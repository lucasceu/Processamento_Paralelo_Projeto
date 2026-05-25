# Processamento Paralelo - Projeto ⚡

Este projeto aplica um filtro Gaussiano em imagens PGM no modo serial e paralelo (OpenMP).

Alunos: Lucas Pereira Céu e Henrique Daniel Resende

## Arquivos .c 📄

- Entrega_01/serial.c: versão serial do filtro Gaussiano.
- Entrega_01/paralelo.c: versão paralela do filtro Gaussiano (OpenMP).

## Requisitos ✅

- GCC com suporte a OpenMP

## Compilar 🛠️

```bash
cd Entrega_01
gcc -O2 -fopenmp -lm serial.c -o serial
gcc -O2 -fopenmp -lm paralelo.c -o paralelo
```

## Executar ▶️

Serial:

```bash
./serial image_2048x2048.pgm 3
```

Paralelo:

```bash
./paralelo image_2048x2048.pgm 3 8
```

O programa imprime o tempo de execução no terminal. Exemplo:

```text
Tempo Paralelo (8 threads): 0.251901 segundos
```

Onde:
- argumento 1: caminho da imagem PGM
- argumento 2: tamanho do kernel (k)
- argumento 3 (paralelo): número de threads

Saídas 📦:

- saida_serial_k<k>.pgm
- saida_paralela_k<k>.pgm
