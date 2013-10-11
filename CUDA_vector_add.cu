//http://www.cs.usfca.edu/~peter/cs625/code/vec_add.cu

#include <stdio.h>
#include <stdlib.h>

/* Kernel for vector addition */
__global__ void Vec_add(float x[], float y[], float output[], int numberOfElements) {
   /* First block gets first threads_per_block components.      */
   /* Second block gets next threads_per_block components, etc. */
   int i = blockDim.x * blockIdx.x + threadIdx.x;

   /* block_count*threads_per_block may be >= n */
   if (i < n){
     output[i] = x[i] + y[i];
   }
}  /* Vec_add */


/* Host code */
int main(int argc, char* argv[]) {
   int n, i;
   float *h_x, *h_y, *h_z;
   float *d_x, *d_y, *d_z;
   int threads_per_block;
   int block_count;
   size_t size;

   /* Get number of components in vector */
   if (argc != 2) {
      fprintf(stderr, "usage: %s <vector order>\n", argv[0]);
      exit(0);
   }
   n = strtol(argv[1], NULL, 10);
   size = n*sizeof(float);

   /* Allocate input vectors in host memory */
   h_x = (float*) malloc(size);
   h_y = (float*) malloc(size);
   h_z = (float*) malloc(size);
   
   /* Initialize input vectors */
   for (i = 0; i < n; i++) {
      h_x[i] = i+1;
      h_y[i] = n-i;
   }

   //================================================================

   /* Allocate vectors in device memory */
   cudaMalloc(&d_x, size);
   cudaMalloc(&d_y, size);
   cudaMalloc(&d_z, size);

   /* Copy vectors from host memory to device memory */
   cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

   /* Define block size */
   threads_per_block = 256;

   /* Define grid size.  The calculation is a kludge to get  */
   /* Ceiling(n/threads_per_block).  This insures we have at */
   /* least one thread per vector component.                 */
   block_count = (n + threads_per_block - 1)/threads_per_block;

   /* Invoke kernel using block_count blocks, each of which  */
   /* contains threads_per_block threads                     */
   Vec_add<<<block_count, threads_per_block>>>(d_x, d_y, d_z, n);

   /* Copy result from device memory to host memory */
   /* h_z contains the result in host memory        */
   cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);

   printf("The sum is: \n");
   for (i = 0; i < n; i++)
      printf("%.1f ", h_z[i]);
   printf("\n");

   /* Free device memory */
   cudaFree(d_x);
   cudaFree(d_y);
   cudaFree(d_z);

   /* Free host memory */
   free(h_x);
   free(h_y);
   free(h_z);

   return 0;
}  /* main */