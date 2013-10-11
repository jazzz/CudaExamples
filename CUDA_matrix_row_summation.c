template< typename T >
__global__ void kRowSum(T* vec, int rowCount, int vecCount,T* out, int offset)
{
	int coldIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int rowIndex = blockIdx.y + offset;
	__shared__ T sum_vec[256];

  // Initialize Shared Memory
	sum_vec[threadIdx.x] = ((colIndex < colCount) ?  vec[vecIndex+ rowIndex*colCount]: 0);

	int halfWidth = blockDim.x/2;
	while(halfWidth > 0)
	{
		__syncthreads();
		if(threadIdx.x < halfWidth){
			sum_vec[threadIdx.x] = sum_vec[threadIdx.x]+ sum_vec[threadIdx.x+ halfWidth]; 
    }
		halfWidth /=2;
	}



	__syncthreads();

	if(threadIdx.x  < colCount)
	{
		out[colIndex + rowIndex*colCount] = sum_vec[threadIdx.x];

	}

}