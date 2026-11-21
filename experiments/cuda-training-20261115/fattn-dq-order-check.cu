#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cstdio>
#include <vector>
#include <cmath>
using namespace nvcuda;
template<int splits, bool strided>
__global__ void check_gemm(const half * a, const half * b, float * c, int m, int n, int k) {
    const int row=blockIdx.x*16, col=blockIdx.y*16;
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::col_major> af;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
    wmma::fragment<wmma::accumulator,16,16,16,float> cf[splits];
    for(int s=0;s<splits;++s)wmma::fill_fragment(cf[s],0.0f);
    for(int i=0;i<k;i+=16) {
        if(i<k-1280)continue;
        const int s=strided?(i/16)%splits: i/(k/splits);
        wmma::load_matrix_sync(af,a+row+i*m,m);
        wmma::load_matrix_sync(bf,b+col*k+i,k);
        wmma::mma_sync(cf[s],af,bf,cf[s]);
    }
    for(int s=1;s<splits;++s)for(int j=0;j<cf[0].num_elements;++j)cf[0].x[j]+=cf[s].x[j];
    wmma::store_matrix_sync(c+row+col*m,cf[0],m,wmma::mem_col_major);
}
template<int splits,bool strided>
void run(const half *a,const half *b,float *output,const float *reference,int m,int n,int k) {
    check_gemm<splits,strided><<<dim3(m/16,n/16),32>>>(a,b,output,m,n,k);
    std::vector<float>d(m*n);cudaMemcpy(d.data(),output,m*n*4,cudaMemcpyDeviceToHost);
    int different=0;double error=0,norm=0;for(int i=0;i<m*n;++i){different+=reference[i]!=d[i];error+=double(reference[i]-d[i])*(reference[i]-d[i]);norm+=double(reference[i])*reference[i];}
    printf("splits=%d strided=%d different=%d relative_l2=%g\n",splits,strided,different,sqrt(error/norm));
}
int main() {
    for(int k : {16384}) {
        int m=256,n=256;
        std::vector<half>a(m*k),b(n*k);
        for(size_t i=0;i<a.size();++i)a[i]=__float2half(float(int(i*37%257)-128)/257);
        for(size_t i=0;i<b.size();++i)b[i]=__float2half((i%k)<size_t(k-1280)?0.0f:float(int(i*17%113)-56)/129);
        half *da,*db;float *dc,*dd;cudaMalloc(&da,a.size()*2);cudaMalloc(&db,b.size()*2);cudaMalloc(&dc,m*n*4);cudaMalloc(&dd,m*n*4);
        cudaMemcpy(da,a.data(),a.size()*2,cudaMemcpyHostToDevice);cudaMemcpy(db,b.data(),b.size()*2,cudaMemcpyHostToDevice);
        cublasHandle_t h;cublasCreate(&h);float alpha=1,beta=0;
        cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,da,CUDA_R_16F,m,db,CUDA_R_16F,k,&beta,dc,CUDA_R_32F,m,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        std::vector<float>c(m*n);cudaMemcpy(c.data(),dc,m*n*4,cudaMemcpyDeviceToHost);
#define RUN(S) run<S,false>(da,db,dd,c.data(),m,n,k);run<S,true>(da,db,dd,c.data(),m,n,k);
        RUN(1) RUN(2) RUN(4) RUN(8) RUN(16)
#undef RUN
        cudaFree(da);cudaFree(db);cudaFree(dc);cudaFree(dd);cublasDestroy(h);
    }
}
