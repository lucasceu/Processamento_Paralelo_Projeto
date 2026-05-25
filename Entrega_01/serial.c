#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef struct {
    int width;
    int height;
    int stride;
    float *data;
} Image;

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

static void pad_from_raw(Image *img, const unsigned char *raw, int half) {
    int p_w = img->stride;
    int p_h = img->height + 2 * half;

    for (int i = 0; i < p_h; i++) {
        for (int j = 0; j < p_w; j++) {
            int src_i = i - half;
            int src_j = j - half;
            if (src_i < 0) src_i = 0;
            if (src_i >= img->height) src_i = img->height - 1;
            if (src_j < 0) src_j = 0;
            if (src_j >= img->width) src_j = img->width - 1;
            img->data[i * p_w + j] = (float)raw[src_i * img->width + src_j];
        }
    }
}

static void apply_blur(Image *src, Image *dst, const float *kernel, int k_size) {
    int half = k_size / 2;
    int p_w = src->stride;

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

static void write_png_from_padded(const char *filename, Image *img, int half) {
    int w = img->width;
    int h = img->height;
    int stride = img->stride;
    unsigned char *out = (unsigned char *)malloc(w * h);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float val = img->data[(i + half) * stride + (j + half)];
            if (val < 0.0f) val = 0.0f;
            if (val > 255.0f) val = 255.0f;
            out[i * w + j] = (unsigned char)val;
        }
    }

    stbi_write_png(filename, w, h, 1, out, w);
    free(out);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: %s <arquivo_imagem> <iteracoes>\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];
    int iterations = (argc > 2) ? atoi(argv[2]) : 10;
    int k_size = 3;
    float sigma = 1.0f;
    int half = k_size / 2;

    int width, height, channels;
    unsigned char *raw = stbi_load(filename, &width, &height, &channels, 1);
    if (!raw) {
        printf("Erro ao carregar a imagem.\n");
        return 1;
    }

    int p_w = width + 2 * half;
    int p_h = height + 2 * half;

    Image img1 = {width, height, p_w, (float *)malloc(p_w * p_h * sizeof(float))};
    Image img2 = {width, height, p_w, (float *)malloc(p_w * p_h * sizeof(float))};
    float *kernel = (float *)malloc(k_size * k_size * sizeof(float));

    pad_from_raw(&img1, raw, half);
    create_gaussian_kernel(kernel, k_size, sigma);

    Image *src = &img1;
    Image *dst = &img2;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int it = 0; it < iterations; it++) {
        apply_blur(src, dst, kernel, k_size);
        Image *tmp = src; src = dst; dst = tmp;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    write_png_from_padded("saida_blur_sequencial.png", src, half);

    printf("Benchmark Sequencial - Arquivo: %s | Iteracoes: %d | Tempo: %.4f s\n",
           filename, iterations, time_taken);

    stbi_image_free(raw);
    free(img1.data);
    free(img2.data);
    free(kernel);
    return 0;
}