/* Kernel for SAXPY addition */
__global__ void Saxpy(float A[], float X[], float Y[], float output[], int numberOfElements) {
   /* First block gets first threads_per_block components.      */
   /* Second block gets next threads_per_block components, etc. */
   int i = blockDim.x * blockIdx.x + threadIdx.x;

   /* block_count*threads_per_block may be >= n */
   if (i < n){
     output[i] = A[i]*X[i] + Y[i];
   }
}  

int main(int argc, char* argv[]) {
   
   //Host Arrays
   size_t size = 1<<24;
   float *h_x, *h_y, *h_a, *h_out;
   float *d_x, *d_y, *d_a, *d_out;
   
   // Initialize Host Arrays
   .....
   
   int byteSize = size * sizeof(float));
   
   //Allocate Device Memory
   cudaMalloc(&d_x, byteSize);
   cudaMalloc(&d_y, byteSize);
   cudaMalloc(&d_a, byteSize);
   cudaMalloc(&d_out, byteSize);
  
   // Mem Copy in
   cudaMemcpy(d_x,h_x,byteSize, cudaMemcpyHostToDevice);
   cudaMemcpy(d_y,h_y,byteSize, cudaMemcpyHostToDevice);
   cudaMemcpy(d_a,h_a,byteSize, cudaMemcpyHostToDevice);
  
   int threadsPerBlock = 256;
   int blockCount = (size-1)/threadsPerBlock +1
   Saxpy<<< blockCount, threadsPerBlock>>>(d_x,d_y,d_a,d_out);
   
   //Mem copy Out
    cudaMemcpy(h_out,d_out,byteSize, cudaMemcpyDeviceToHost);
  
   // Deallocate Device Arrays;
   cudaFree(d_x);
   cudaFree(d_y);
   cudaFree(d_a);
   cudaFree(d_out);
  }


