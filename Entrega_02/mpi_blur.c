/*
 * mpi_blur.c — Desfoque Gaussiano iterativo com MPI (memória distribuída)
 * Etapa 2 do projeto de Processamento Paralelo.
 *
 * Compilação:
 *   mpicc -O2 -Wall -o mpi_blur mpi_blur.c -lm
 *
 * Uso:
 *   mpirun -np <P> ./mpi_blur <imagem> <iteracoes> <k_size>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef struct {
    int width;
    int height;
    int stride;  // width + 2*half
    float *data;
} Image;

// Cria o kernel gaussiano normalizado — mesma fórmula do serial.c
static void create_gaussian_kernel(float *kernel, int size, float sigma) {
    int half = size / 2;
    float sum = 0.0f;
    for (int i = -half; i <= half; i++) {
        for (int j = -half; j <= half; j++) {
            float val = expf(-(i * i + j * j) / (2.0f * sigma * sigma));
            kernel[(i + half) * size + (j + half)] = val;
            sum += val;
        }
    }
    for (int i = 0; i < size * size; i++) kernel[i] /= sum;
}

// Preenche o buffer com padding clamp a partir dos pixels raw
// Linhas/colunas fora da imagem replicam a borda mais próxima
static void pad_from_raw(Image *img, const unsigned char *raw, int half) {
    int p_w = img->stride;
    int p_h = img->height + 2 * half;
    for (int i = 0; i < p_h; i++) {
        for (int j = 0; j < p_w; j++) {
            int src_i = i - half;
            int src_j = j - half;
            if (src_i < 0)            src_i = 0;
            if (src_i >= img->height) src_i = img->height - 1;
            if (src_j < 0)            src_j = 0;
            if (src_j >= img->width)  src_j = img->width  - 1;
            img->data[i * p_w + j] = (float)raw[src_i * img->width + src_j];
        }
    }
}

// Blur gaussiano — idêntico ao serial.c: loops i→j→ki→kj, mesma indexação
static void apply_blur(Image *src, Image *dst, const float *kernel, int k_size) {
    int half = k_size / 2;
    int p_w  = src->stride;
    for (int i = half; i < src->height + half; i++) {
        for (int j = half; j < src->width + half; j++) {
            float sum = 0.0f;
            for (int ki = -half; ki <= half; ki++) {
                for (int kj = -half; kj <= half; kj++) {
                    sum += src->data[(i + ki) * p_w + (j + kj)] *
                           kernel[(ki + half) * k_size + (kj + half)];
                }
            }
            dst->data[i * p_w + j] = sum;
        }
    }
}

// Salva o buffer com padding como PNG, descartando as bordas de padding
static void write_png_from_padded(const char *filename, Image *img, int half) {
    int w = img->width;
    int h = img->height;
    int s = img->stride;
    unsigned char *out = (unsigned char *)malloc((size_t)w * h);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float val = img->data[(i + half) * s + (j + half)];
            if (val < 0.0f)   val = 0.0f;
            if (val > 255.0f) val = 255.0f;
            out[i * w + j] = (unsigned char)val;
        }
    }
    stbi_write_png(filename, w, h, 1, out, w);
    free(out);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int my_rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 2) {
        if (my_rank == 0)
            fprintf(stderr, "Uso: %s <imagem> <iteracoes> <k_size>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    char *filename = argv[1];
    int iterations = (argc > 2) ? atoi(argv[2]) : 10;
    int k_size     = (argc > 3) ? atoi(argv[3]) : 3;

    // Rank 0 carrega a imagem; os demais recebem width/height via Bcast logo depois
    int width = 0, height = 0;
    unsigned char *raw = NULL;

    if (my_rank == 0) {
        int channels;
        raw = stbi_load(filename, &width, &height, &channels, 1);
        if (!raw) {
            fprintf(stderr, "Rank 0: erro ao carregar '%s'\n", filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Só width, height, k_size e iterations precisam ser transmitidos
    // O kernel é recriado localmente por cada processo (é determinístico)
    MPI_Bcast(&width,      1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height,     1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k_size,     1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int half   = k_size / 2;
    int stride = width  + 2 * half;
    int p_h    = height + 2 * half;

    // Rank 0 aloca os dois buffers completos e preenche o padding
    float *full_buf_A = NULL, *full_buf_B = NULL;
    if (my_rank == 0) {
        full_buf_A = (float *)malloc((size_t)p_h * stride * sizeof(float));
        full_buf_B = (float *)malloc((size_t)p_h * stride * sizeof(float));
        if (!full_buf_A || !full_buf_B) {
            fprintf(stderr, "Rank 0: falha de alocação dos buffers completos\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        Image img_full = {width, height, stride, full_buf_A};
        pad_from_raw(&img_full, raw, half);
        stbi_image_free(raw);
        raw = NULL;
    }

    // Divide as linhas da imagem entre os processos
    // Usa height % nprocs para distribuir o resto uma linha por vez
    int *lines_per_rank = (int *)malloc(nprocs * sizeof(int));
    int *offset_lines   = (int *)malloc(nprocs * sizeof(int));
    for (int r = 0; r < nprocs; r++)
        lines_per_rank[r] = height / nprocs + (r < height % nprocs ? 1 : 0);
    offset_lines[0] = 0;
    for (int r = 1; r < nprocs; r++)
        offset_lines[r] = offset_lines[r - 1] + lines_per_rank[r - 1];

    int local_lines    = lines_per_rank[my_rank];
    int local_buf_rows = local_lines + 2 * half;  // dados reais + halos superior e inferior

    // Cada processo aloca dois buffers locais para ping-pong
    // calloc garante zeros nas áreas de halo antes da inicialização
    float *local_data_A = (float *)calloc((size_t)local_buf_rows * stride, sizeof(float));
    float *local_data_B = (float *)calloc((size_t)local_buf_rows * stride, sizeof(float));
    if (!local_data_A || !local_data_B) {
        fprintf(stderr, "Rank %d: falha de alocação dos buffers locais\n", my_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Monta os vetores de contagem e deslocamento para o Scatterv
    // sendcounts[r] = quantos floats o rank r vai receber
    // displs_v[r]   = offset dentro do sendbuf (que começa na linha half do buffer completo)
    int *sendcounts = (int *)malloc(nprocs * sizeof(int));
    int *displs_v   = (int *)malloc(nprocs * sizeof(int));
    for (int r = 0; r < nprocs; r++) {
        sendcounts[r] = lines_per_rank[r] * stride;
        displs_v[r]   = offset_lines[r]   * stride;
    }

    // Scatter das linhas reais (pula as half linhas de padding do topo no sendbuf)
    // No recvbuf cada processo escreve a partir de half*stride para reservar espaço ao halo
    MPI_Scatterv(
        my_rank == 0 ? full_buf_A + half * stride : NULL,
        sendcounts, displs_v, MPI_FLOAT,
        local_data_A + half * stride,
        local_lines * stride, MPI_FLOAT,
        0, MPI_COMM_WORLD
    );

    // Inicializa os halos externos por clamp (replica a linha de borda)
    // Rank 0 não tem vizinho acima; último rank não tem vizinho abaixo
    // Essas bordas ficam constantes durante todo o loop
    if (half > 0) {
        if (my_rank == 0) {
            for (int h = 0; h < half; h++)
                memcpy(local_data_A + (size_t)h * stride,
                       local_data_A + (size_t)half * stride,
                       stride * sizeof(float));
        }
        if (my_rank == nprocs - 1) {
            for (int h = 0; h < half; h++)
                memcpy(local_data_A + (size_t)(half + local_lines + h) * stride,
                       local_data_A + (size_t)(half + local_lines - 1) * stride,
                       stride * sizeof(float));
        }
    }

    // Copia o estado inicial (com halos de borda) para o segundo buffer de ping-pong
    memcpy(local_data_B, local_data_A, (size_t)local_buf_rows * stride * sizeof(float));

    // Cada processo cria seu próprio kernel — é determinístico para o mesmo k_size
    float *kernel = (float *)malloc(k_size * k_size * sizeof(float));
    create_gaussian_kernel(kernel, k_size, 1.0f);

    // Image.height = local_lines para que apply_blur itere só sobre as linhas desta fatia
    Image local_img_A = {width, local_lines, stride, local_data_A};
    Image local_img_B = {width, local_lines, stride, local_data_B};
    Image *local_src  = &local_img_A;
    Image *local_dst  = &local_img_B;

    // tag_down: mensagem de rank r para rank r+1
    // tag_up:   mensagem de rank r para rank r-1
    const int tag_down = 1;
    const int tag_up   = 2;

    double t_comp  = 0.0;
    double t_comm  = 0.0;

    // Timer começa imediatamente após o Scatterv (operação coletiva já sincroniza)
    double t_inicio = MPI_Wtime();

    for (int it = 0; it < iterations; it++) {

        // Troca de halos — ordering par/ímpar para evitar deadlock
        //
        // Layout do buffer local:
        //   [0 .. half-1]                           halo superior
        //   [half .. half+local_lines-1]             dados reais
        //   [half+local_lines .. 2*half+local_lines-1]  halo inferior
        //
        // Bloco descendente — pares (0,1), (2,3), (4,5)...
        // Cada rank envia suas últimas half linhas reais para rank+1 (viram halo superior de rank+1)
        // e recebe as primeiras half linhas de rank+1 (viram seu halo inferior)
        double tc0 = MPI_Wtime();

        if (my_rank % 2 == 0) {
            if (my_rank + 1 < nprocs)
                MPI_Send(local_src->data + local_lines * stride,
                         half * stride, MPI_FLOAT,
                         my_rank + 1, tag_down, MPI_COMM_WORLD);
            if (my_rank + 1 < nprocs)
                MPI_Recv(local_src->data + (half + local_lines) * stride,
                         half * stride, MPI_FLOAT,
                         my_rank + 1, tag_up, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            if (my_rank - 1 >= 0)
                MPI_Recv(local_src->data,
                         half * stride, MPI_FLOAT,
                         my_rank - 1, tag_down, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (my_rank - 1 >= 0)
                MPI_Send(local_src->data + half * stride,
                         half * stride, MPI_FLOAT,
                         my_rank - 1, tag_up, MPI_COMM_WORLD);
        }

        // Bloco ascendente — pares (1,2), (3,4), (5,6)...
        // Complementa o bloco anterior: cobre os pares que ficaram de fora
        if (my_rank % 2 == 0) {
            if (my_rank - 1 >= 0)
                MPI_Recv(local_src->data,
                         half * stride, MPI_FLOAT,
                         my_rank - 1, tag_down, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (my_rank - 1 >= 0)
                MPI_Send(local_src->data + half * stride,
                         half * stride, MPI_FLOAT,
                         my_rank - 1, tag_up, MPI_COMM_WORLD);
        } else {
            if (my_rank + 1 < nprocs)
                MPI_Send(local_src->data + local_lines * stride,
                         half * stride, MPI_FLOAT,
                         my_rank + 1, tag_down, MPI_COMM_WORLD);
            if (my_rank + 1 < nprocs)
                MPI_Recv(local_src->data + (half + local_lines) * stride,
                         half * stride, MPI_FLOAT,
                         my_rank + 1, tag_up, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        t_comm += MPI_Wtime() - tc0;

        // Blur local sobre as linhas desta fatia (com halos já atualizados)
        double tp0 = MPI_Wtime();
        apply_blur(local_src, local_dst, kernel, k_size);
        t_comp += MPI_Wtime() - tp0;

        Image *tmp = local_src;
        local_src  = local_dst;
        local_dst  = tmp;
    }

    // Timer encerra antes do Gatherv
    double t_fim   = MPI_Wtime();
    double t_total = t_fim - t_inicio;

    // gather_buf é dedicado ao resultado MPI — separado dos buffers de validação sequencial
    // Assim full_buf_A e full_buf_B ficam disponíveis para o ping-pong sequencial logo abaixo
    float *gather_buf = NULL;
    if (my_rank == 0)
        gather_buf = (float *)malloc((size_t)p_h * stride * sizeof(float));

    // Gatherv primeiro — todos os ranks participam sem esperar rank 0 terminar outra coisa
    // Cada rank envia suas linhas reais (sem halos); rank 0 monta a imagem completa em gather_buf
    MPI_Gatherv(
        local_src->data + half * stride,
        local_lines * stride, MPI_FLOAT,
        my_rank == 0 ? gather_buf + half * stride : NULL,
        sendcounts, displs_v, MPI_FLOAT,
        0, MPI_COMM_WORLD
    );

    // Validação sequencial (rank 0 apenas) — roda depois do Gatherv
    // Reutiliza full_buf_A (entrada original, intocada) e full_buf_B como ping-pong
    // O resultado sequencial fica em seq_src->data; compara contra gather_buf
    if (my_rank == 0) {
        memcpy(full_buf_B, full_buf_A, (size_t)p_h * stride * sizeof(float));

        Image seq_img_A = {width, height, stride, full_buf_A};
        Image seq_img_B = {width, height, stride, full_buf_B};
        Image *seq_src  = &seq_img_A;
        Image *seq_dst  = &seq_img_B;

        for (int it = 0; it < iterations; it++) {
            apply_blur(seq_src, seq_dst, kernel, k_size);
            Image *tmp = seq_src; seq_src = seq_dst; seq_dst = tmp;
        }

        // Salva a imagem de saída MPI
        Image img_mpi_result = {width, height, stride, gather_buf};
        write_png_from_padded("saida_blur_mpi.png", &img_mpi_result, half);

        // Diferença máxima absoluta pixel a pixel entre MPI e sequencial
        float delta_max = 0.0f;
        for (int i = half; i < height + half; i++) {
            for (int j = half; j < width + half; j++) {
                float diff = fabsf(gather_buf[i * stride + j]
                                 - seq_src->data[i * stride + j]);
                if (diff > delta_max) delta_max = diff;
            }
        }

        double comm_pct = (t_total > 0.0) ? (t_comm / t_total * 100.0) : 0.0;
        const char *status = (delta_max < 1e-3f) ? "OK" : "FALHOU";

        printf("Benchmark MPI - Arquivo: %s | Kernel: %dx%d | Processos: %d | Iteracoes: %d\n",
               filename, k_size, k_size, nprocs, iterations);
        printf("  Tempo Total:        %.4f s\n", t_total);
        printf("  Tempo Computacao:   %.4f s  [tempo local do rank 0]\n", t_comp);
        printf("  Tempo Comunicacao:  %.4f s  [tempo local do rank 0] (Overhead: %.1f%%)\n",
               t_comm, comm_pct);
        printf("  Corretude: %s (delta_max = %.2e)\n", status, (double)delta_max);

        // CSV em append — escreve cabeçalho só se o arquivo estiver vazio
        FILE *csv = fopen("resultados_mpi.csv", "a");
        if (csv) {
            fseek(csv, 0, SEEK_END);
            if (ftell(csv) == 0)
                fprintf(csv, "Imagem,Kernel,Iteracoes,Processos,"
                             "Tempo_Total_s,Tempo_Comp_s,Tempo_Comm_s,"
                             "Delta_Max,Corretude\n");
            fprintf(csv, "%s,%d,%d,%d,%.6f,%.6f,%.6f,%.2e,%s\n",
                    filename, k_size, iterations, nprocs,
                    t_total, t_comp, t_comm, (double)delta_max, status);
            fclose(csv);
        }

        free(gather_buf);
        free(full_buf_A);
        free(full_buf_B);
    }

    free(local_data_A);
    free(local_data_B);
    free(kernel);
    free(lines_per_rank);
    free(offset_lines);
    free(sendcounts);
    free(displs_v);

    MPI_Finalize();
    return 0;
}
