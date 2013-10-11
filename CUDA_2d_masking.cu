/* Kernel for SAXPY addition */
__global__ void 2D_Masking(int A[], int output[], int threshold, int sizeX, intsizeY) {
   /* First block gets first threads_per_block components.      */
   /* Second block gets next threads_per_block components, etc. */
   int _x = blockDim.x * blockIdx.x + threadIdx.x; 
   int _y = blockDim.y * blockIdx.y + threadIdx.y; 
   
   /* block_count*threads_per_block may be >= n */
   if (_x < sizeX && _y < sizeY){
      output[i] = (A[_y * sizeX + _x] > threshold ) 1 : 0 ; 
   }
}  

int main(int argc, char* argv[]) {
   
   //Host Arrays
   size_t size = 1024*800;
   float *h_x, *h_out;
   float *d_x, *d_out;
   
   // Initialize Host Arrays
   .....
   
   int byteSize = size * sizeof(float));
   
   //Allocate Device Memory
   cudaMalloc(&d_x, byteSize);
   cudaMalloc(&d_out, byteSize);
  
   // Mem Copy in
   cudaMemcpy(d_x,h_x,byteSize, cudaMemcpyHostToDevice);
  
   int threadsPerBlock = 256;
   int blockCount = (size-1)/threadsPerBlock +1
   2D_Masking<<< blockCount, threadsPerBlock>>>(d_x,67,d_out);
   
   //Mem copy Out
    cudaMemcpy(h_out,d_out,byteSize, cudaMemcpyDeviceToHost);
  
   // Deallocate Device Arrays;
   cudaFree(d_x);
   cudaFree(d_out);
  }


