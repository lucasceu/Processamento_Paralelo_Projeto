# Processamento Paralelo - Projeto

Implementação do filtro Gaussiano iterativo em quatro versões: serial, paralela com OpenMP, paralela com MPI e paralela em GPU com CUDA.

Alunos: Lucas Pereira Céu e Henrique Daniel Resende

---

## Ambiente de Hardware

Ambiente utilizado para compilação e execução dos benchmarks.

### Sistema

| Componente | Especificação |
|------------|---------------|
| Notebook |
| Sistema operacional | Microsoft Windows 11 Home Single Language |
| Ambiente de execução | WSL2 — Ubuntu 24.04 (kernel Linux 6.6.87) |
| Memória RAM (host) | 16 GB (15,4 GB utilizáveis) |
| Memória RAM (WSL) | 7,4 GB alocados |

### Processador (CPU)

| Componente | Especificação |
|------------|---------------|
| Modelo | AMD Ryzen 5 5600H with Radeon Graphics |
| Arquitetura | x86_64 |
| Núcleos físicos | 6 |
| Threads lógicos | 12 |

### Placa de vídeo (GPU)

| Componente | Especificação |
|------------|---------------|
| Modelo | NVIDIA GeForce RTX 3050 Laptop GPU |
| Memória de vídeo (VRAM) | 4096 MB (4 GB) |
| Streaming Multiprocessors (SMs) | 16 |
| Compute Capability | 8.6 |
| Driver NVIDIA | 596.08 |
| Versão CUDA (driver) | 13.2 |

### Software e ferramentas

| Ferramenta | Versão |
|------------|--------|
| GCC | 13.3.0 (Ubuntu) |
| CUDA Toolkit (`nvcc`) | 12.0 |
| OpenMPI | 4.1.6 |
| Compilador MPI | `mpicc` (OpenMPI) |

---

## Estrutura

```
Entrega_01/          # Etapa 1 — Serial e OpenMP
  serial.c
  paralelo.c

Entrega_02/          # Etapa 2 — MPI
  serial.c           # baseline serial (para medir speedup)
  mpi_blur.c         # implementação MPI
  stb_image.h
  stb_image_write.h

Entrega_03/   # Etapa 3 — CUDA
  cuda_blur.cu       # implementação CUDA (global + shared memory)
  run_benchmarks.sh  # script de benchmark automatizado
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

---

## Etapa 3 — CUDA

Paralelização massiva em GPU: cada thread CUDA calcula um pixel de saída. Duas variantes:
- **Global** (modo 0): leitura direta da memória global do device.
- **Shared** (modo 1, padrão): carrega tiles na `__shared__` memory dos SMs, reduzindo acessos à DRAM da GPU.

O kernel Gaussiano é armazenado em `__constant__` memory para acesso eficiente em broadcast.

### Requisitos
- `nvcc` (CUDA Toolkit ≥ 11) disponível no PATH
- GPU NVIDIA compatível com CUDA

### Compilar

```bash
cd Entrega_03
gcc -O2 -c stb_impl.c -o stb_impl.o
nvcc -O2 -o cuda_blur cuda_blur.cu stb_impl.o -lm
```

### Executar

```bash
# Shared memory (padrão, modo 1)
./cuda_blur base1024x1024.png 100 3

# Memória global (modo 0)
./cuda_blur base1024x1024.png 100 3 0

# Script completo de benchmark (gera resultados_cuda.csv)
chmod +x run_benchmarks.sh
./run_benchmarks.sh
```

Argumentos: `<imagem> <iterações> <k_size> [modo]` (k_size: `3` ou `5`, modo: `0`=global, `1`=shared)

### Saídas

- `saida_blur_cuda.png` — imagem processada pela GPU
- `resultados_cuda.csv` — métricas (tempo GPU, tempo serial, speedup, corretude)

---

## Declaração de Transparência e Uso de IA

Declaramos que este projeto contou com o auxílio das ferramentas de IA Google Gemini e Claude como recurso de apoio exclusivamente para as tarefas de:

- Auxílio na estruturação de trechos de código e na identificação de gargalos ou erros de sintaxe nos scripts desenvolvidos;
- Orientação na busca de bibliotecas e ferramentas correlatas para suporte à implementação do benchmark;
- Esclarecimento de dúvidas pontuais sobre conceitos de computação paralela e primitivas de programação concorrente;
- Revisão gramatical de trechos do relatório.

Como autores, atestamos que revisamos, testamos e validamos criticamente todo o conteúdo gerado, assumindo total e exclusiva responsabilidade pela correção lógica do código, precisão dos relatórios de desempenho e integridade acadêmica do material entregue.

Henrique Daniel Resende e Lucas Pereira Céu - 03 de julho de 2026

