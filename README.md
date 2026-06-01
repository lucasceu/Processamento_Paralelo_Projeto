# Processamento Paralelo - Projeto

Implementação do filtro Gaussiano iterativo em três versões: serial, paralela com OpenMP e paralela com MPI.

Alunos: Lucas Pereira Céu e Henrique Daniel Resende

---

## Estrutura

```
Entrega_01/          # Etapa 1 — Serial e OpenMP
  serial.c
  paralelo.c

Entrega_02/          # Etapa 2 — MPI
  serial.c           # baseline serial (para medir speedup)
  mpi_blur.c         # implementação MPI
  relatorio.html     # relatório com resultados e gráficos
  stb_image.h
  stb_image_write.h
```

---

## Etapa 1 — Serial e OpenMP

### Requisitos
- GCC com suporte a OpenMP

### Compilar

```bash
cd Entrega_01
gcc -O2 serial.c -o serial -lm
gcc -O2 -fopenmp paralelo.c -o paralelo -lm
```

### Executar

```bash
./serial base1024x1024.png 100 5
OMP_NUM_THREADS=8 ./paralelo base1024x1024.png 100 5
```

Argumentos: `<imagem> <iterações> <kernel>` (kernel: `3` ou `5`)

Saídas: `saida_blur_sequencial.png`, `saida_blur_paralela.png`

---

## Etapa 2 — MPI

Paralelização via decomposição de domínio 1D: a imagem é dividida em faixas horizontais distribuídas entre os processos MPI. A cada iteração, processos vizinhos trocam bordas (halo exchange) para calcular pixels de fronteira corretamente.

### Requisitos
- GCC + OpenMPI (`mpicc`, `mpirun`)
- Biblioteca stb_image (já inclusa na pasta)

### Compilar

```bash
cd Entrega_02
mpicc -O2 -o serial serial.c -lm
mpicc -O2 -o mpi_blur mpi_blur.c -lm
```

### Executar

```bash
# Baseline serial (para calcular speedup)
./serial base1024x1024.png 100 5

# MPI com N processos
mpirun -np 4 ./mpi_blur base1024x1024.png 100 5

# Se o ambiente não tiver slots suficientes (ex: WSL)
mpirun --oversubscribe -np 8 ./mpi_blur base1024x1024.png 100 5
```

Argumentos: `<imagem> <iterações> <kernel>` (kernel: `3` ou `5`)

### Saídas

- `saida_blur_mpi.png` — imagem processada pelo MPI
- `resultados_mpi.csv` — métricas de desempenho (tempo total, computação, comunicação, corretude)

O programa valida automaticamente a corretude comparando o resultado MPI contra uma execução serial no rank 0 e reporta o delta máximo absoluto pixel a pixel.

### Relatório

Abra `Entrega_02/relatorio.html` no navegador para ver a análise completa com tabelas e gráficos interativos (tempos, speedup, eficiência, overhead de comunicação, comparação OpenMP vs MPI).
