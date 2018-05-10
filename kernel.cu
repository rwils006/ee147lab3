/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE

__global__ void kernel_Hist(unsigned int* array, unsigned int size, unsigned int* histo,unsigned int num_bins){
		
	__shared__ unsigned int histo_private[2048];

	if(threadIdx.x<num_bins)histo_private[threadIdx.x] = 0;
	__syncthreads();


	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int stride = blockDim.x*gridDim.x; //total number of threads
	
	while(i<size){
		atomicAdd(&(histo_private[array[i]]),1);
		i+=stride;
	}

	__syncthreads();
	
	if(threadIdx.x<num_bins){
		atomicAdd(&(histo[threadIdx.x]),histo_private[threadIdx.x]);
	}


}
/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

	dim3 block(num_bins*num_bins);
	dim3 grid((num_elements + block.x - 1)/block.x);

	
	kernel_Hist<<<grid,block>>>(input,num_elements,bins,num_bins);


}


