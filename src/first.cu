#include <stdio.h>
#include <chrono>
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
        __host__ __device__ Queue(
            int * data,
            int size,
            int totalSize,
            int tid
        );
        __device__ __host__ void insert(Job job);
        __device__ __host__ bool empty();
        __device__ __host__ Job pop();
};


// The kernel that handles jobs
__global__ 
void handleJobs(
    int * inputData,
    int * inputSizes,
    int * outputData,
    int * outputSizes,
    int maxLength
);

// the job handler in the cpu 
void cpuHandleJobs(
    int * inputData,
    int * inputSizes,
    int * outputData,
    int * outputSizes,
    int * params 
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

//returns the maxSize and modifies other arrays in place
void create(
    int * input,
    int * inputSizes,
    int * params
); 

int main(){
    // We setup variables, divisions are for the number of parallel jobs in gpu
    int divisions = 2048;
    int maxJobsPerJob = 3;
    int maxJobSize = 4;
    
    // Max size as we don't allocate when handling jobs 
    int maxSize = divisions * maxJobsPerJob * maxJobSize * 5;
    int params[] = {divisions,maxJobsPerJob,maxJobSize,maxSize}; 

    std::cout << "Divisions " << divisions << std::endl;
    std::cout << "Max Jobs per Job " << maxJobsPerJob << std::endl;
    std::cout << "Max Jobs size " << maxJobSize << std::endl;
    std::cout << "Max Size " << maxSize << std::endl;

    // Declare arrays input, output and sizes 
    int * inputCPU = new int[maxSize]; 
    int * outputCPU = new int[maxSize]; 
    int * inputSizesCPU = new int[divisions]; 
    int * outputSizesCPU = new int[divisions]; 


    // Declare GPU arrays
    int * inputGPU;
    int * outputGPU;
    int * inputSizesGPU;
    int * outputSizesGPU; 

    // std::cout << "About to create " << maxSize << std::endl;
    // We create random jobs with random inputs 
    // The function will divide on the number of jobs, with variable sizes
    // but with space left over
    create(inputCPU,inputSizesCPU,&params[0]);

    // std::cout << "creation sucessful " << maxSize << std::endl;
    // debugging
    /*
    for(int i=0;i<maxSize;i++){
        std::cout << inputCPU[i];
    }
    std::cout << std::endl;
    for(int i=0;i<divisions;i++){
        std::cout << inputSizesCPU[i] << " ";
    }
    std::cout << std::endl;
    */
    // Set outputs to 0s
    memset(outputCPU,0,sizeof(int)*maxSize);
    memset(outputSizesCPU,0,sizeof(int)*divisions);

    // Mallocs for gpu arrays
    /*std::cout << " "<< */ cudaMalloc(&inputGPU,maxSize*sizeof(int));
    /*std::cout << " "<< */cudaMalloc(&outputGPU,maxSize*sizeof(int));
    /*std::cout << " "<< */ cudaMalloc(&inputSizesGPU,divisions*sizeof(int));
    /*std::cout << " "<< */ cudaMalloc(&outputSizesGPU,divisions*sizeof(int));
    //std::cout << std::endl;

    // Copying data to GPU 
    /*std::cout << " "<< */cudaMemcpy(
        inputGPU,
        inputCPU,
        maxSize*sizeof(int),
        cudaMemcpyHostToDevice
    );
    /*std::cout << " "<<  */cudaMemcpy(
        inputSizesGPU,
        inputSizesCPU,
        divisions*sizeof(int),
        cudaMemcpyHostToDevice
    );
    //std::cout << std::endl;


    // Clearing output arrays 
    /*std::cout << " "<< */ cudaMemset(outputGPU,0,maxSize*sizeof(int));
    /*std::cout << " "<< */ cudaMemset(outputSizesGPU,0,divisions*sizeof(int));
   // std::cout << std::endl;

    // We setup events to measure
    cudaEvent_t start, end;
    /*std::cout << " "<<*/  cudaEventCreate(&start);
    /*std::cout << " "<<*/  cudaEventCreate(&end);
    //std::cout << std::endl;

    // We run the kernel and measure
    //std::cout << "kernel launched " << cudaGetLastError() <<std::endl;
    cudaEventRecord(start);
    handleJobs<<<1,divisions>>>(
        inputGPU,
        inputSizesGPU,
        outputGPU,
        outputSizesGPU,
        maxSize/divisions
    );
    //std::cout << "kernel finished " << cudaGetLastError() << std::endl;
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // Print time ellapsed
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    std::cout << "Gpu Test "<<milliseconds << " ms" << std::endl;
    
    // We run the cpujob and measure time

    auto start_cpu = std::chrono::high_resolution_clock::now();

    cpuHandleJobs(inputCPU,inputSizesCPU,outputCPU,outputSizesCPU,params);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end_cpu - start_cpu;

    printf("Cpu Test: %f ms \n",duration.count());
    // We reduce the sizes for comparison

    //std::cout << "cpu finished"<< std::endl;
    /*
    for(int i=0;i<maxSize;i++){
        std::cout << inputCPU[i];
    }
    std::cout << std::endl;
    for(int i=0;i<divisions;i++){
        std::cout << inputSizesCPU[i] << " ";
    }
    std::cout << std::endl;
    for(int i=0;i<maxSize;i++){
        std::cout << outputCPU[i];
    }
    std::cout << std::endl;
    for(int i=0;i<divisions;i++){
        std::cout << outputSizesCPU[i] << " ";
    }
    */
    // We compare

    int * resultSizeCPU = new int[divisions];

    cudaDeviceSynchronize();

    int resultSizeReducedCPU = thrust::reduce(
            outputSizesCPU,outputSizesCPU+divisions-1);
    // std::cout << "resultsReduced CPU "<< resultSizeReducedCPU << std::endl;

    //We copy the size of the results back to cpu
	cudaMemcpy(
        resultSizeCPU,
        outputSizesGPU,
        divisions*sizeof(int),
        cudaMemcpyDeviceToHost
    );
    int resultSizeReducedGPU = thrust::reduce(
            resultSizeCPU,resultSizeCPU+divisions-1);

    int * outputGPU2 = new int[maxSize];
	cudaMemcpy(
        outputGPU2,
        outputGPU,
        maxSize*sizeof(int),
        cudaMemcpyDeviceToHost
    );
    cudaDeviceSynchronize();
    /*
    std::cout << std::endl;
    for(int i=0;i<maxSize;i++){
        std::cout << outputGPU2[i];
    }
    std::cout << std::endl;
    for(int i=0;i<divisions;i++){
        std::cout << resultSizeCPU[i] << " ";
    }
    */

    // std::cout << "outputObtained "<< std::endl;
    if(resultSizeReducedCPU==resultSizeReducedGPU){
        int * compareKernelArray; 
        int * compareKernelArray2 = new int[maxSize]; 
        int * outputCPU2; 

        cudaMalloc(&compareKernelArray,sizeof(int)*maxSize);
        cudaMalloc(&outputCPU2,sizeof(int)*maxSize);
        /* std::cout << " "<<  */cudaMemcpy(
            outputCPU2,
            outputCPU,
            maxSize*sizeof(int),
            cudaMemcpyHostToDevice
        );
        cudaMemset(compareKernelArray,0,sizeof(int)*maxSize);
        compare<<<1,maxSize>>>
            (outputCPU2,outputGPU,compareKernelArray,maxSize);

        cudaMemcpy(
            compareKernelArray2,
            compareKernelArray,
            sizeof(int)*maxSize,
            cudaMemcpyDeviceToHost
        );
        /*
        std::cout << std::endl;
        for(int i=0;i<maxSize;i++){
            std::cout << outputGPU2[i];
        }
        std::cout << std::endl;
        for(int i=0;i<maxSize;i++){
            std::cout << outputCPU[i];
        }
        std::cout << std::endl;
        for(int i=0;i<maxSize;i++){
            std::cout << compareKernelArray2[i] << " ";
        }
        */
        //std::cout << std::endl;

        int result = thrust::reduce(
            compareKernelArray2,
            compareKernelArray2+resultSizeReducedCPU-1
        );

        // We reduce and check if it equals 0
        if(!result)
            // We print the results
            std::cout<< "Results were equal" << std::endl;
        else{
            std::cout << result << std::endl; 
            // We print the results
            std::cout<<"Results weren't equal"<<std::endl;
        }
        delete compareKernelArray2; 
        cudaFree(outputCPU2); 
        cudaFree(compareKernelArray); 
    }
    else{
        std::cout<<"Results were not equal"<<std::endl;
    } 
    delete resultSizeCPU;
    delete outputGPU2;
    delete inputCPU; 
    delete outputCPU; 
    delete inputSizesCPU; 
    delete outputSizesCPU; 
    cudaFree(inputGPU);
    cudaFree(outputGPU);
    cudaFree(inputSizesGPU);
    cudaFree(outputSizesGPU);
    return 0;
}

// Here we handle our jobs and save the results that are outputted
__global__ 
void handleJobs(
    int * inputData,
    int * inputSizes,
    int * outputData,
    int * outputSizes,
    int maxSize
){
    int tid = threadIdx.x;
    Queue * jobs = new Queue(inputData,inputSizes[tid],maxSize,tid);
    Queue * results = new Queue(outputData,outputSizes[tid],maxSize,tid);
    while(!jobs->empty()){
        if( jobs->start > jobs->totalSize &&
        results->end > results->totalSize){
                break;
        }
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
    outputSizes[tid]=results->end;
}

void cpuHandleJobs(
    int * inputData,
    int * inputSizes,
    int * outputData,
    int * outputSizes,
    int * params 
){
    int divisions = params[0];
    int maxLength = params[3];
    for(int i=0;i<divisions;i++){
        Queue * jobs = new Queue(
            inputData,
            inputSizes[i],
            maxLength/divisions,
            i
        );
        Queue * results = new Queue(
            outputData,
            outputSizes[i],
            maxLength/divisions,
            i
        );

        //std::cout << "start " << jobs->start 
        //    <<" end "<< jobs->end << std::endl;

        while(!jobs->empty()){
            if( jobs->start > jobs->totalSize &&
            results->end > results->totalSize){
                std::cout << "Se lleno la cola"<< std::endl;
                std::cout << "jobs "<< jobs->end<< " "<< 
                    jobs->totalSize<<std::endl;
                std::cout << "results "<< results->end<< " "<< 
                    results->totalSize<<std::endl;
            }

            Job job = jobs->pop();
            /*
            std::cout << "job being processed " << job.getType() 
                << std::endl;
            std::cout << "start " << jobs->data[jobs->start] 
                <<" end "<< jobs->data[jobs->end-1] << std::endl;
            std::cout << "job 1 " << job.data[1] << std::endl;
            std::cout << "job 2 " << job.data[2] << std::endl;
            std::cout << "job 3 " << job.data[3] << std::endl;
            std::cout << "start " << jobs->start <<" end "<< jobs->end 
                << std::endl;
            //sleep(1);*/
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
        outputSizes[i]=results->end;
    }
}

void create(int * input,int * inputSizes,int * params){
    int divisions = params[0];
    int maxJobsPerJob = params[1];
    int maxSizePerJob = params[2];
    int maxSize = params[3];
    
    srand(time(NULL));
    // Divide max size between divisions maxSizePerJob and maxJobsPerJob to declare upper bound, lower bound is that amount halved
    int lowerBound = maxSize/maxJobsPerJob/maxSizePerJob/divisions/2;
    // std::cout << " lowerBound " << lowerBound << std::endl;
    if(lowerBound==0){
        std::cout << " Too small maxSize set " << std::endl;
        exit(1);
    }
    for(int i = 0;i<divisions;i++){
        int start = i*maxSize/divisions;
        int numberOfJobs = rand() % lowerBound + lowerBound; 
        for(int j=0;j<numberOfJobs;j++){
            int type  = rand() % 3 + 1; 
            input[start+inputSizes[i]] = type;
            for(int k=1;k!=type;k++){
                input[start+inputSizes[i]+k]= rand() % 9 + 1;
            }
            inputSizes[i]+=1+type;
        }
    }
}

__global__ void compare(
  int * array1,
  int * array2,
  int * output,
  int n){
    int tid = threadIdx.x;
    if(tid < n){
        output[tid] = array1[tid]==array2[tid];
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
__device__ __host__ Queue::Queue(
    int * data,
    int size,
    int totalSize,
    int tid
){
    this->data = data + tid * totalSize;
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
