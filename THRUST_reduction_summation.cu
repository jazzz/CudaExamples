  #include <thrust/count.h>
  #include <thrust/device_vector.h>
  
  int main(){
  
  // put three 1s in a device_vector
  thrust::device_vector<int> vec(5,0);
  vec[1] = 1;
  vec[3] = 1;
  vec[4] = 1;
  
  // count the 1s
  int result = thrust::count(vec.begin(), vec.end(), 1);
  // result is three
  
  return result;
  }