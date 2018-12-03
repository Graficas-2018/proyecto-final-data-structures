#include <stdio.h>
#include <unistd.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <cstdlib>
#include <time.h>
#include <iostream>

// The queues for the simple jobs has a host and a device implementation
class Job{
    public:
        int * data;
        __host__ __device__ Job(int * array);
        __host__ __device__ int getType();
        __host__ __device__ int * getData();
};

class Queue{
    public:
        int start;
        int end;
        int * data;
        int totalSize;
        __device__ Queue(
            int * data,
            int size,
            int totalSize,
            int tid
        );
        __host__  Queue(int * data,int size, int totalSize);
        __device__ __host__ void insert(Job job);
        __device__ __host__ bool empty();
        __device__ __host__ Job pop();
};


// The kernel that handles jobs
__global__ 
void handleJobs(
    int * inputData,
    int size,
    int maxSize,
    int * resultSize,
    int * outputData
);

// the job handler in the cpu 
int cpuHandleJobs(
    int * inputData,
    int inputLength,
    int * outputData,
    int maxLength
);

// The host and device jobs
__device__ __host__ Job job1(Job job,Queue * jobqueue,Queue * results);
__device__ __host__ Job job2(Job job,Queue * jobqueue,Queue * results);
__device__ __host__ Job job3(Job job,Queue * jobqueue,Queue * results);

// The random generation of data and jobs will be done with thrust in the main

// a map reduce will be used for the comparison of results
__global__ void compare(
  int * array1,
  int * array2,
  int * output,
  int n
);


int main(){
    // We create random jobs with random inputs in 32 different sections of the same array, in order to compare the results in cpu and gpu
    //int size = rand() % (32*5);
    int size = 32;
    std::cout << "size set to " << size << std::endl;

    // Max size as we don't allocate when handling jobs 
    int maxSize = 10*size;

    // Declare vectors
    int * inputCPU = new int[maxSize]; 
    int * outputCPU = new int[maxSize]; 
    int * outputCPU2 = new int[maxSize]; 
    int * inputGPU;
    int * outputGPU;

    cudaMalloc(&inputGPU,maxSize*sizeof(int));
    cudaMalloc(&outputGPU,maxSize*sizeof(int));
    int totalSize = 0;
    for(int i=0;i<size;i++){
        inputCPU[i*7+0]=3;
        inputCPU[i*7+1]=2;
        inputCPU[i*7+2]=3;
        inputCPU[i*7+3]=4;
        inputCPU[i*7+4]=2;
        inputCPU[i*7+5]=2;
        inputCPU[i*7+6]=3;
        totalSize+=7;
    }
    for(int i=0;i<totalSize;i++){
        std::cout << inputCPU[i];
    }
    std::cout<<std::endl;
    for(int i=0;i<totalSize;i++){
        std::cout << outputCPU[i];
    }
    std::cout<<std::endl;
    for(int i=0;i<totalSize;i++){
        std::cout << outputCPU2[i];
    }
    maxSize = totalSize*3;

    std::cout << std::endl << "jobs generated " << totalSize << std::endl;
    cudaMemcpy(inputCPU,&inputGPU,size*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemset(&outputGPU,-1,maxSize*sizeof(int));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    int * resultSize;
    cudaMalloc((void**)&resultSize,sizeof(int)*32);

    // We run the kernel and measure
    cudaEventRecord(start);
    std::cout << "kernel launched " << size << std::endl;
    handleJobs<<<32,1>>>(
        inputGPU,
        totalSize/32,
        maxSize,
        resultSize,
        outputGPU 
    );
    std::cout << "kernel finished " << std::endl;
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "time ellapsed "<<milliseconds << std::endl;
    
    // We run the cpujob and measure
    int cpuSize = cpuHandleJobs(inputCPU,totalSize,outputCPU,maxSize);
    std::cout << "cpu finished, cpuSize "<< cpuSize << std::endl;

    for(int i=0;i<totalSize;i++){
        std::cout << inputCPU[i];
    }
    std::cout << std::endl;
    for(int i=0;i<totalSize;i++){
        std::cout << outputCPU[i];
    }
    // We compare

    int * resultSizeCPU = new int[32];
	cudaMemcpy(resultSize,&resultSizeCPU,32,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << "resultsObtained "<< std::endl;
    int resultSizeReduced = thrust::reduce(resultSizeCPU,resultSizeCPU+31);
    cudaDeviceSynchronize();
    std::cout << "resultsReduced "<< resultSizeReduced << std::endl;
    int * outputGPU2 = new int[resultSizeReduced];
	cudaMemcpy(outputGPU,&outputGPU2,resultSizeReduced,
            cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << "outputObtained "<< std::endl;

    /*if(cpuSize==resultSizeReduced){
        int * compareKernelArray = new int(cpuSize);
        compare<<<cpuSize,1>>>
            (outputCPU,outputGPU2,compareKernelArray,cpuSize);

        // We reduce and check if it equals 0
        if(thrust::reduce(compareKernelArray,compareKernelArray+cpuSize)==0)
            // We print the results
            std::cout<< "Results were equal" << std::endl;
        else
            // We print the results
            std::cout<<"Results were not equal"<<std::endl;
    }
    else{
        std::cout<<"Results were not equal"<<std::endl;
    } */
    return 0;
}

// Here we handle our jobs and save the results that are outputted
__global__ 
void handleJobs(
    int * inputData,
    int size,
    int maxSize,
    int * resultSize,
    int * outputData
){
    int tid = threadIdx.x;
    Queue * jobs = new Queue(inputData,size,maxSize,tid);
    Queue * results = new Queue(outputData,*resultSize,maxSize,tid);
    while(!jobs->empty()){
        Job job = jobs->pop();
        switch(job.getType()){
            case 1:
                job1(job,jobs,results);
            break;
            case 2:
                job2(job,jobs,results);
            break;
            case 3:
                job3(job,jobs,results);
            break;
        } 
    }
    resultSize[tid]=results->end;
}

int cpuHandleJobs(
    int * inputData,
    int inputLength,
    int * outputData,
    int maxLength
){
    Queue * jobs = new Queue(inputData,inputLength,maxLength);
    Queue * results = new Queue(outputData,0,maxLength);

    std::cout << "start " << jobs->start <<" end "<< jobs->end << std::endl;

    while(!jobs->empty() 
        && jobs->end < maxLength 
        && results->end < maxLength){

        Job job = jobs->pop();

        std::cout << "job being processed " << job.getType() << std::endl;
    std::cout << "start " << jobs->data[jobs->start] 
        <<" end "<< jobs->data[jobs->end-1] << std::endl;
        std::cout << "job 1 " << job.data[1] << std::endl;
        std::cout << "job 2 " << job.data[2] << std::endl;
        std::cout << "job 3 " << job.data[3] << std::endl;
        std::cout << "start " << jobs->start <<" end "<< jobs->end 
            << std::endl;
        //sleep(1);
        switch(job.getType()){
            case 1:
                job1(job,jobs,results);
            break;
            case 2:
                job2(job,jobs,results);
            break;
            case 3:
                job3(job,jobs,results);
            break;
        } 
    }
    return results->end;
}

__global__ void compare(
  int * array1,
  int * array2,
  int * output,
  int n){
    int tid = threadIdx.x;
    if(tid < n){
        output[tid] = (int) (array1[tid]==array2[tid]);
    }
    else output[tid] = 0;
}
__device__ __host__ Job::Job(int * data){
    this->data = data;
}

__device__ int Job::getType(){
    return this->data[0];
}
__device__ int * Job::getData(){
    return this->data+1;
}
__device__ Queue::Queue(
    int * data,
    int size,
    int totalSize,
    int tid
){
    this->data = data + tid*size;
    this->start = 0;
    this->end = this->start + size;
    this->totalSize = totalSize;
}
__host__ Queue::Queue(int * data,int size,int totalSize){
    this->data = data;
    this->start = 0;
    this->end = size;
    this->totalSize = totalSize;
}
__device__ __host__ void Queue::insert(Job job){
    this->end += 1+job.getType();
}
__device__ __host__ bool Queue::empty(){
    return start==end;
}
__device__ __host__ Job Queue::pop(){
    Job job = Job(data+this->start);
    this->start += 1+job.getType();
    return job;
}

__device__ __host__ Job job3(Job job,Queue * jobqueue,Queue * results){
    int result = 0;
    if(job.data[3]%2==0){
        result = job.data[2]+job.data[1];
    }
    else{
        result = job.data[2]-job.data[1];
    }
    // insert new job to be reprocessed
    jobqueue->data[jobqueue->end] = 2;
    jobqueue->data[jobqueue->end+1] = result;
    jobqueue->data[jobqueue->end+2] = job.data[3];
    Job njob = Job(jobqueue->data+jobqueue->end);
    jobqueue->insert(njob);
    // insert into results too 
    results->data[results->end] = 2;
    results->data[results->end+1] = result;
    results->data[results->end+2] = job.data[3];
    Job njob2 = Job(results->data+results->end);
    results->insert(njob2);
    return 0;
}
__device__ __host__ Job job2(Job job,Queue * jobqueue,Queue * results){
    int result = 0;
    result = 2*job.data[1]-job.data[2];

    // insert new job to be reprocessed
    jobqueue->data[jobqueue->end] = 1;
    jobqueue->data[jobqueue->end+1] = result;
    Job njob = Job(jobqueue->data+jobqueue->end);
    jobqueue->insert(njob);

    // insert into results too 
    results->data[results->end] = 1;
    results->data[results->end+1] = result;
    Job njob2 = Job(results->data+results->end);
    results->insert(njob2);
    return 0;
}
__device__ __host__ Job job1(Job job,Queue * jobqueue,Queue * results){
    results->data[results->end++] = 1;
    results->data[results->end++] = 2;
    return 0;
}
