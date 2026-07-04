/*
 * cuda_blur.cu — Desfoque Gaussiano iterativo com CUDA (Etapa 3)
 * DEC107 - Processamento Paralelo
 *
 * Compilacao:
 *   gcc -O2 -c stb_impl.c -o stb_impl.o
 *   nvcc -O2 -o cuda_blur cuda_blur.cu stb_impl.o -lm
 *
 * Uso: ./cuda_blur <imagem> <iteracoes> <k_size> [modo]
 *      modo: 0=global, 1=shared(padrao)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

/* Forward declarations das funcoes STB (implementacao em stb_impl.c) */
typedef unsigned char stbi_uc;
extern "C" stbi_uc *stbi_load(const char *f, int *x, int *y, int *ch, int req);
extern "C" void     stbi_image_free(void *p);
extern "C" int      stbi_write_png(const char *f, int w, int h, int comp, const void *data, int stride);





#define CUDA_CHECK(call) \
    do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
        fprintf(stderr,"CUDA error %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE);} } while(0)

__constant__ float d_gauss[25];
#define TILE_W 16
#define TILE_H 16

/* ---- Kernel 1: memoria global ---- */
__global__ void blur_global(const float *src, float *dst,
                             int width, int height, int stride, int k_size)
{
    int tx=blockIdx.x*blockDim.x+threadIdx.x;
    int ty=blockIdx.y*blockDim.y+threadIdx.y;
    if(tx>=width||ty>=height) return;
    int half=k_size/2, pi=ty+half, pj=tx+half;
    float s=0.0f;
    for(int ki=0;ki<k_size;ki++)
        for(int kj=0;kj<k_size;kj++)
            s+=src[(pi-half+ki)*stride+(pj-half+kj)]*d_gauss[ki*k_size+kj];
    dst[pi*stride+pj]=s;
}

/* ---- Kernel 2: memoria compartilhada (tiled) ---- */
__global__ void blur_shared(const float *src, float *dst,
                             int width, int height, int stride, int k_size)
{
    extern __shared__ float smem[];
    const int half=k_size/2;
    const int smem_w=TILE_W+2*half, smem_h=TILE_H+2*half;
    const int p_h=height+2*half;
    const int lx=threadIdx.x, ly=threadIdx.y;
    const int r0=blockIdx.y*TILE_H, c0=blockIdx.x*TILE_W;
    const int bsz=TILE_W*TILE_H, tot=smem_w*smem_h, tid=ly*TILE_W+lx;

    for(int idx=tid;idx<tot;idx+=bsz){
        int sy=idx/smem_w, sx=idx%smem_w;
        int gy=r0+sy, gx=c0+sx;
        if(gy>=p_h) gy=p_h-1;
        if(gx>=stride) gx=stride-1;
        smem[sy*smem_w+sx]=src[gy*stride+gx];
    }
    __syncthreads();

    int oi=half+blockIdx.y*TILE_H+ly;
    int oj=half+blockIdx.x*TILE_W+lx;
    if(oi<height+half && oj<width+half){
        float s=0.0f;
        for(int ki=0;ki<k_size;ki++)
            for(int kj=0;kj<k_size;kj++)
                s+=smem[(ly+ki)*smem_w+(lx+kj)]*d_gauss[ki*k_size+kj];
        dst[oi*stride+oj]=s;
    }
}

/* ---- Auxiliares host ---- */
static void make_kernel(float *k,int sz,float sig){
    int h=sz/2; float sum=0;
    for(int i=-h;i<=h;i++) for(int j=-h;j<=h;j++){
        float v=expf(-(i*i+j*j)/(2*sig*sig));
        k[(i+h)*sz+(j+h)]=v; sum+=v;
    }
    for(int i=0;i<sz*sz;i++) k[i]/=sum;
}
static void pad(float *buf,const unsigned char *raw,int w,int h,int stride,int half){
    int ph=h+2*half,pw=stride;
    for(int i=0;i<ph;i++) for(int j=0;j<pw;j++){
        int si=i-half,sj=j-half;
        if(si<0)si=0; if(si>=h)si=h-1;
        if(sj<0)sj=0; if(sj>=w)sj=w-1;
        buf[i*pw+j]=(float)raw[si*w+sj];
    }
}
static void blur_cpu(const float *src,float *dst,int w,int h,int stride,int ksz,const float *k){
    int half=ksz/2;
    for(int i=half;i<h+half;i++) for(int j=half;j<w+half;j++){
        float s=0;
        for(int ki=-half;ki<=half;ki++) for(int kj=-half;kj<=half;kj++)
            s+=src[(i+ki)*stride+(j+kj)]*k[(ki+half)*ksz+(kj+half)];
        dst[i*stride+j]=s;
    }
}
static void save_png(const char *f,const float *buf,int w,int h,int stride,int half){
    unsigned char *out=(unsigned char*)malloc((size_t)w*h);
    for(int i=0;i<h;i++) for(int j=0;j<w;j++){
        float v=buf[(i+half)*stride+(j+half)];
        if(v<0)v=0; if(v>255)v=255;
        out[i*w+j]=(unsigned char)v;
    }
    stbi_write_png(f,w,h,1,out,w); free(out);
}

/* ---- main ---- */
int main(int argc,char *argv[]){
    if(argc<2){fprintf(stderr,"Uso: %s <img> <iter> <k> [modo]\n",argv[0]);return 1;}
    const char *fn=argv[1];
    int iter=(argc>2)?atoi(argv[2]):100;
    int ksz =(argc>3)?atoi(argv[3]):3;
    int shrd=(argc>4)?atoi(argv[4]):1;
    if(ksz!=3&&ksz!=5){fprintf(stderr,"k_size: 3 ou 5\n");return 1;}
    const char *mode=shrd?"shared":"global";
    int half=ksz/2;

    int W,H,ch;
    unsigned char *raw=stbi_load(fn,&W,&H,&ch,1);
    if(!raw){fprintf(stderr,"Erro: %s\n",fn);return 1;}

    int stride=W+2*half, ph=H+2*half;
    size_t nb=(size_t)stride*ph*sizeof(float);

    float *hA=(float*)malloc(nb), *hR=(float*)malloc(nb), *kh=(float*)malloc(ksz*ksz*4);
    pad(hA,raw,W,H,stride,half);
    make_kernel(kh,ksz,1.0f);
    stbi_image_free(raw);

    float *dA,*dB;
    CUDA_CHECK(cudaMalloc(&dA,nb)); CUDA_CHECK(cudaMalloc(&dB,nb));
    CUDA_CHECK(cudaMemcpy(dA,hA,nb,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hA,nb,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(d_gauss,kh,ksz*ksz*sizeof(float)));

    dim3 block(TILE_W,TILE_H);
    dim3 grid((W+TILE_W-1)/TILE_W,(H+TILE_H-1)/TILE_H);
    size_t smem=(size_t)(TILE_W+2*half)*(TILE_H+2*half)*sizeof(float);

    cudaEvent_t e0,e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

    float *ds=dA,*dd=dB;
    CUDA_CHECK(cudaEventRecord(e0));
    for(int it=0;it<iter;it++){
        if(shrd) blur_shared<<<grid,block,smem>>>(ds,dd,W,H,stride,ksz);
        else     blur_global<<<grid,block>>>(ds,dd,W,H,stride,ksz);
        float *t=ds; ds=dd; dd=t;
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());

    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1));
    double tg=ms/1000.0;

    CUDA_CHECK(cudaMemcpy(hR,ds,nb,cudaMemcpyDeviceToHost));
    save_png("saida_blur_cuda.png",hR,W,H,stride,half);

    /* Serial para validacao e speedup */
    float *sA=(float*)malloc(nb),*sB=(float*)malloc(nb);
    memcpy(sA,hA,nb); memcpy(sB,hA,nb);
    float *ss=sA,*sd=sB;
    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int it=0;it<iter;it++){
        blur_cpu(ss,sd,W,H,stride,ksz,kh);
        float *t=ss; ss=sd; sd=t;
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double tc=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    float dmax=0;
    for(int i=half;i<H+half;i++) for(int j=half;j<W+half;j++){
        float d=fabsf(hR[i*stride+j]-ss[i*stride+j]);
        if(d>dmax)dmax=d;
    }
    double spd=(tg>0)?tc/tg:0;
    const char *st=(dmax<1e-3f)?"OK":"FALHOU";

    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));

    printf("Benchmark CUDA | %s | %dx%d | iter=%d | modo=%s\n",fn,ksz,ksz,iter,mode);
    printf("  GPU:                %s\n",prop.name);
    printf("  SMs:                %d\n",prop.multiProcessorCount);
    printf("  Mem Global:         %.0f MB\n",prop.totalGlobalMem/(1024.0*1024.0));
    printf("  Tempo GPU:          %.4f s\n",tg);
    printf("  Tempo Serial (CPU): %.4f s\n",tc);
    printf("  Speedup:            %.2fx\n",spd);
    printf("  Corretude:          %s (delta_max=%.2e)\n",st,(double)dmax);
    printf("  Grade:              (%u x %u) blocos de (%u x %u) threads\n",grid.x,grid.y,block.x,block.y);
    if(shrd) printf("  Shared mem/bloco:   %zu bytes\n",smem);

    FILE *csv=fopen("resultados_cuda.csv","a");
    if(csv){
        fseek(csv,0,SEEK_END);
        if(ftell(csv)==0)
            fprintf(csv,"Imagem,Kernel,Iteracoes,Modo,Tempo_GPU_s,Tempo_Serial_s,Speedup,Delta_Max,Corretude,GPU\n");
        fprintf(csv,"%s,%d,%d,%s,%.6f,%.6f,%.4f,%.2e,%s,%s\n",
                fn,ksz,iter,mode,tg,tc,spd,(double)dmax,st,prop.name);
        fclose(csv);
    }

    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaEventDestroy(e0)); CUDA_CHECK(cudaEventDestroy(e1));
    free(hA);free(hR);free(sA);free(sB);free(kh);
    return 0;
}
