#include <iostream>
int main(){
    // We create the random jobs with random inputs
    //int size = rand() % (32*5);
    int size = 32;
    std::cout << "size set to " << size << std::endl;
    // Max size as we don't allocate in memory
    int maxSize = size*10;

    // Declare vectors
    int * inputCPU = new int[maxSize]; 
    int * outputCPU = new int[maxSize]; 
    int * outputCPU2 = new int[maxSize]; 
    int totalSize = 0;
    for(int i=0;i<size;i++){
        /*inputCPU[i*7+0]=3;
        inputCPU[i*7+1]=2;
        inputCPU[i*7+2]=3;
        inputCPU[i*7+3]=4;
        inputCPU[i*7+4]=2;
        inputCPU[i*7+5]=2;
        inputCPU[i*7+6]=3;
        totalSize+=7;*/
        inputCPU[i*2+0]=1;
        inputCPU[i*2+1]=1;
        totalSize+=2;
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
    std::cout << std::endl << "jobs generated " << totalSize << std::endl;
}
