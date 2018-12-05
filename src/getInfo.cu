
#include <stdio.h>
#include <math.h>
int main(){

    // GetGPUProperties
    cudaDeviceProp props; 
    cudaGetDeviceProperties(&props,0);

    // Get maximum threads, blocks and grids
    printf("GPU Info\n");
    printf("Name: %s\n",props.name);
    printf("Max Threads Per Block  %d\n",props.maxThreadsPerBlock);
    printf("Max Threads Size  %d %d %d\n",
        props.maxThreadsDim[0],
        props.maxThreadsDim[1],
        props.maxThreadsDim[2]);
    printf("Max Grid Size %d %d %d\n",
        props.maxGridSize[0],
        props.maxGridSize[1],
        props.maxGridSize[2]);
    printf("Compute Capability %d\n",props.major);
}
