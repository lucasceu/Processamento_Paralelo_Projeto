#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Função para gerar o kernel Gaussiano
double* generate_gaussian_kernel(int k, double sigma) {
    double* kernel = (double*) malloc(k * k * sizeof(double));
    double sum = 0.0;
    int half = k / 2;
    for (int i = -half; i <= half; i++) {
        for (int j = -half; j <= half; j++) {
            double val = exp(-(i * i + j * j) / (2.0 * sigma * sigma));
            kernel[(i + half) * k + (j + half)] = val;
            sum += val;
        }
    }
    for (int i = 0; i < k * k; i++) kernel[i] /= sum;
    return kernel;
}

// Leitura de imagem no formato PGM (P5 binário ou P2 texto)
double* read_pgm(const char *filename, int *width, int *height) {
    FILE *file = fopen(filename, "rb");
    if (!file) { printf("Erro ao abrir %s\n", filename); exit(1); }
    
    char magic[3];
    int max_val;
    fscanf(file, "%2s", magic);
    
    // Ignora comentários
    int c = fgetc(file);
    while (c == '#' || c == '\n' || c == '\r' || c == ' ') {
        if (c == '#') while ((c = fgetc(file)) != '\n' && c != EOF);
        else c = fgetc(file);
    }
    ungetc(c, file);
    
    fscanf(file, "%d %d %d", width, height, &max_val);
    fgetc(file); // consome o \n após max_val

    double *image = (double*) malloc((*width) * (*height) * sizeof(double));
    
    if (magic[1] == '5') { // Binário
        unsigned char *temp = (unsigned char*) malloc((*width) * (*height));
        fread(temp, 1, (*width) * (*height), file);
        for (int i = 0; i < (*width) * (*height); i++) image[i] = (double)temp[i];
        free(temp);
    } else { // Texto
        int val;
        for (int i = 0; i < (*width) * (*height); i++) {
            fscanf(file, "%d", &val);
            image[i] = (double)val;
        }
    }
    fclose(file);
    return image;
}

// Escrita de imagem no formato PGM
void write_pgm(const char *filename, double *image, int width, int height) {
    FILE *file = fopen(filename, "wb");
    fprintf(file, "P5\n%d %d\n255\n", width, height);
    unsigned char *temp = (unsigned char*) malloc(width * height);
    for (int i = 0; i < width * height; i++) {
        double val = image[i];
        if (val > 255.0) val = 255.0;
        if (val < 0.0) val = 0.0;
        temp[i] = (unsigned char)val;
    }
    fwrite(temp, 1, width * height, file);
    free(temp);
    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: %s <imagem.pgm> <tamanho_kernel>\n", argv[0]);
        return 1;
    }

    char *input_file = argv[1];
    int k = atoi(argv[2]);
    int iterations = 10;
    int width, height;

    double *image = read_pgm(input_file, &width, &height);
    double *result = (double*) malloc(width * height * sizeof(double));
    for (int i = 0; i < width * height; i++) result[i] = image[i];

    double *kernel = generate_gaussian_kernel(k, 1.0);
    int half_k = k / 2;

    double start_time = omp_get_wtime();

    for (int iter = 0; iter < iterations; iter++) {
        for (int i = half_k; i < height - half_k; i++) {
            for (int j = half_k; j < width - half_k; j++) {
                double sum = 0.0;
                for (int ki = -half_k; ki <= half_k; ki++) {
                    for (int kj = -half_k; kj <= half_k; kj++) {
                        sum += image[(i + ki) * width + (j + kj)] * kernel[(ki + half_k) * k + (kj + half_k)];
                    }
                }
                result[i * width + j] = sum;
            }
        }
        if (iter < iterations - 1) {
            for (int i = 0; i < width * height; i++) image[i] = result[i];
        }
    }

    double end_time = omp_get_wtime();

    char output_file[100];
    sprintf(output_file, "saida_serial_k%d.pgm", k);
    write_pgm(output_file, result, width, height);

    printf("Tempo Serial: %f segundos\n", end_time - start_time);

    free(image); free(result); free(kernel);
    return 0;
}