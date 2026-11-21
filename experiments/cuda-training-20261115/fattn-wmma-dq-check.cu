#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cstdio>
#include <vector>
#include <cmath>
using namespace nvcuda;
__global__ void check_gemm(const half * a, const half * b, float * c, int m, int n, int k) {
    const int row=blockIdx.x*16, col=blockIdx.y*16;
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::col_major> af;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bf;
    wmma::fragment<wmma::accumulator,16,16,16,float> cf;
    wmma::fill_fragment(cf,0.0f);
    for(int i=k-1280;i<k;i+=16) {
        wmma::load_matrix_sync(af,a+row+i*m,m);
        wmma::load_matrix_sync(bf,b+col*k+i,k);
        wmma::mma_sync(cf,af,bf,cf);
    }
    wmma::store_matrix_sync(c+row+col*m,cf,m,wmma::mem_col_major);
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
        check_gemm<<<dim3(m/16,n/16),32>>>(da,db,dd,m,n,k);
        std::vector<float> c(m*n),d(m*n);cudaMemcpy(c.data(),dc,m*n*4,cudaMemcpyDeviceToHost);cudaMemcpy(d.data(),dd,m*n*4,cudaMemcpyDeviceToHost);
        int different=0;double error=0,norm=0;for(int i=0;i<m*n;++i){different+=c[i]!=d[i];error+=double(c[i]-d[i])*(c[i]-d[i]);norm+=double(c[i])*c[i];}
        printf("k=%d different=%d/%d relative_l2=%g error=%s\n",k,different,m*n,sqrt(error/norm),cudaGetErrorString(cudaGetLastError()));
        cudaFree(da);cudaFree(db);cudaFree(dc);cudaFree(dd);cublasDestroy(h);
    }
}
