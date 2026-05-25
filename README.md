# Processamento Paralelo - Projeto

Este projeto aplica um filtro Gaussiano em imagens PGM no modo serial e paralelo (OpenMP).

## Arquivos .c

- Entrega_01/serial.c: versao serial do filtro Gaussiano.
- Entrega_01/paralelo.c: versao paralela do filtro Gaussiano (OpenMP).

## Requisitos

- GCC com suporte a OpenMP

## Compilar

```bash
cd Entrega_01
gcc -O2 -fopenmp -lm serial.c -o serial
gcc -O2 -fopenmp -lm paralelo.c -o paralelo
```

## Executar

Serial:

```bash
./serial image_2048x2048.pgm 3
```

Paralelo:

```bash
./paralelo image_2048x2048.pgm 3 8
```

Onde:
- argumento 1: caminho da imagem PGM
- argumento 2: tamanho do kernel (k)
- argumento 3 (paralelo): numero de threads

Saidas:

- saida_serial_k<k>.pgm
- saida_paralela_k<k>.pgm
