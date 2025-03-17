#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("Your device count is: %i\n", device_count);

    cudaDeviceProp devProp; 
    for(unsigned int i = 0; i < device_count; i++) 
    {
        cudaGetDeviceProperties(&devProp, i);
        printf("Device name: %s\n", devProp.name);
        printf("Max threads per block: %i\n", devProp.maxThreadsPerBlock);
        printf("Number of SM's: %i\n", devProp.multiProcessorCount);
        printf("Clock frequency: %i\n", devProp.clockRate);
        printf("Max thread dim x: %i\n", devProp.maxThreadsDim[0]);
        printf("Max thread dim y: %i\n", devProp.maxThreadsDim[1]);
        printf("Max thread dim z: %i\n", devProp.maxThreadsDim[2]);
        printf("Max block size x: %i\n", devProp.maxGridSize[0]);
        printf("Max block size y: %i\n", devProp.maxGridSize[1]);
        printf("Max block size z: %i\n", devProp.maxGridSize[2]);
        printf("Registers per block: %i\n", devProp.regsPerBlock);
        printf("Registers per SM: %i\n", devProp.regsPerMultiprocessor);
        printf("Max threads per SM: %i\n", devProp.maxThreadsPerMultiProcessor);
        printf("Warp size: %i\n", devProp.warpSize);
        printf("Compute mode: %i\n", devProp.computeMode);
        printf("max blocks per SM: %i\n", devProp.maxBlocksPerMultiProcessor);
        printf("Shared Memory per block: %lu\n", devProp.sharedMemPerBlock);
    }
    return 0;
}