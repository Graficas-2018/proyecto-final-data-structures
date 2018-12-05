# proyecto-final-data-structures

This project focuses in the following problems:
There are no good data structures for use in cuda development 
    Thrust and cudpp focus on algorithms primarily.
GPUs are SIMD (single instruction, multiple data) and not MIMD.
 variable input and variable output are a nightmare in cuda

How do we solve these problems:
- Making good data structures for use on each thread, array based for easy communication between host and gpu
- Creating a generic job system which allows the same instruction with ifs simulate a MIMD
- For variable input on each one, data structures which allow consumtion as long as there is data
- For variable output, a separate data structure, for each thread. Saving all results never overwriting any. and excessive preallocation outside kernels.

In this project we explore the following:
- A queue-like data structure with variable size per member.
- A job system, to simulate MIMD in SIMD.
- Jobs that generate new jobs to be processed, inserted on the queue.
- Variable sized outputs

The program in src/first.cu:
Uses the queue to add jobs and data
Uses a kernel which handles the jobs and the data.
With random data, we generate random jobs and random input data with variable size

## Results
In the following Device:
    GPU Info
    Name: GeForce GT 750M
    Max Threads Per Block  1024
    Max Threads Size  1024 1024 64
    Max Grid Size 2147483647 65535 65535
    Compute Capability 3
With the following variables
    Max Jobs per Job 3
    Max Jobs size 4
    Max Size 3840

# first run
blocks 1 threads 64 
Gpu Test 1.2177 ms
Cpu Test: 0.023006 ms 

blocks 1 threads 128
Gpu Test 0.693408 ms
Cpu Test: 0.040666 ms 

blocks 1 threads 256
Gpu Test 0.984032 ms
Cpu Test: 0.078343 ms 

blocks 1 threads 512
Gpu Test 1.30102 ms
Cpu Test: 0.158102 ms 

blocks 1 threads 1024
Gpu Test 1.93357 ms
Cpu Test: 0.313306 ms 

blocks 2 threads 1024
Gpu Test 2.66886 ms
Cpu Test: 0.620085 ms 

blocks 4 threads 1024
Gpu Test 5.27821 ms
Cpu Test: 1.245803 ms 

blocks 8 threads 1024
Gpu Test 46.7007 ms
Cpu Test: 2.667552 ms 

blocks 16 threads 1024
Gpu Test 49.1333 ms
Cpu Test: 5.007045 ms 
Results were equal

A second less erratic run showed the following

64  |1.031      |0.021769
---------------------------
128 |0.688672   |0.038911
---------------------------
256 |0.90784    |0.07243
---------------------------
512 |1.3        | 0.142722
---------------------------
1024|1.8687     |  0.309221
---------------------------
2048|2.6982     |0.54996
---------------------------
4096|6.07453    |1.2453
---------------------------
8192|16.2346    |2.444411



# Analysis
We can see a lower almost linear growth for GPU before hitting the 2 blocks afterwards we can see a very exponential growth. While the original analysis within one block suggested it could outpace cpu in the long run. The analysis with more blocks shows different. Also we have two very erratic datum at the end of the first run but the second one shows an expontential increase.

Sequential penalties within GPU should have gone lower instead of higher outside of a single block, but they could be caused by a similar optimization process as the one in warps, but at the block level.

# Conclusions
More configurations would have been optimal for the analysis but the code had to be rewritten multiple times and some math operations were bound to fail. Also there are no good indications to continue as the expected positive results have been proven false.

Instead of a lower linear growth,  exponential growth is the most probable

The first run results are in [results](results.txt)
The second run results are in [results](results2.txt)
Graphs can be find in [excel](Graphs.xlsx) 
