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
- For variable output, a separate data structure, for each thread. Saving all results never overwriting any. and allocation within kernels.

In this project we explore the following:
- A queue-like data structure with variable size per member.
- A job system, to simulate MIMD in SIMD.
- Jobs that generate new jobs to be processed, inserted on the queue.
- Variable sized outputs
- IPC, can we use the results generated in other threads?

The first program:
Uses the queue to add jobs and data
Uses a kernel which handles the jobs and the data.
With random data, we generate random jobs and random input data with variable size
We process this in 32 threads and compare with cpu.

The second example:
Attempts at ipc, with a simple reduce sum and reusing our jobs and queue.
In which the kernel processes adds which in this case take input from neighbors.

The third example:
Tries a waiter which allows the queue to be fed on top of what it already has, meaning we make a super long reduce and partially feed it while the original process handles it.
