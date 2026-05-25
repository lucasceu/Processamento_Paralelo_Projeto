# Processamento Paralelo - Projeto ⚡

Este projeto aplica um filtro Gaussiano em imagens PNG/JPG no modo serial e paralelo (OpenMP).

Alunos: Lucas Pereira Céu e Henrique Daniel Resende

## Arquivos .c 📄

- Entrega_01/serial.c: versão serial do filtro Gaussiano.
- Entrega_01/paralelo.c: versão paralela do filtro Gaussiano (OpenMP).

## Requisitos ✅

- GCC com suporte a OpenMP

## Compilar 🛠️

```bash
cd Entrega_01
gcc -O2 -lm serial.c -o serial
gcc -O2 -fopenmp -lm paralelo.c -o paralelo
```

## Executar ▶️

Serial:

```bash
./serial base1024x1024.png 10
```

Paralelo:

```bash
OMP_NUM_THREADS=8 ./paralelo base1024x1024.png 100
```

O programa imprime o tempo de execução no terminal. Exemplo:

```text
Arquivo: base1024x1024.png | Threads: 8 | Iteracoes: 100 | Tempo: 0.2519 s
```

Onde:
- argumento 1: caminho da imagem
- argumento 2: número de iterações (opcional)

Se omitido, o padrão é 10 no serial e 100 no paralelo.

Saídas 📦:

- saida_blur_sequencial.png
- saida_blur_paralela.png
