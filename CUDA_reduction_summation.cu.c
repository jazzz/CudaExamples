int main(void)
{
  // create array of 256k elements
  const int num_elements = 1<<18;

  // generate random input on the host
  std::vector<float> h_input(num_elements);
  for(int i = 0; i < h_input.size(); ++i)
  {
    h_input[i] = random_float();
  }

  const float host_result = std::accumulate(h_input.begin(), h_input.end(), 0.0f);
  std::cerr << "Host sum: " << host_result << std::endl;

  // move input to device memory
  float *d_input = 0;
  cudaMalloc((void**)&d_input, sizeof(float) * num_elements);
  cudaMemcpy(d_input, &h_input[0], sizeof(float) * num_elements, cudaMemcpyHostToDevice);

  const size_t block_size = 512;
  const size_t num_blocks = (num_elements/block_size) + ((num_elements%block_size) ? 1 : 0);

  // allocate space to hold one partial sum per block, plus one additional
  // slot to store the total sum
  float *d_partial_sums_and_total = 0;
  cudaMalloc((void**)&d_partial_sums_and_total, sizeof(float) * (num_blocks + 1));

  // launch one kernel to compute, per-block, a partial sum
  block_sum<<<num_blocks,block_size,block_size * sizeof(float)>>>(d_input, d_partial_sums_and_total, num_elements);

  // launch a single block to compute the sum of the partial sums
  block_sum<<<1,num_blocks,num_blocks * sizeof(float)>>>(d_partial_sums_and_total, d_partial_sums_and_total + num_blocks, num_blocks);

  // copy the result back to the host
  float device_result = 0;
  cudaMemcpy(&device_result, d_partial_sums_and_total + num_blocks, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Device sum: " << device_result << std::endl;

  // deallocate device memory
  cudaFree(d_input);
  cudaFree(d_partial_sums_and_total);

  return 0;
}
