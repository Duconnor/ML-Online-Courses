CS 179: GPU Computing
Assignment 2

PART 1

Question 1.1: Latency Hiding (5 points)
---------------------------------------


Question 1.2: Thread Divergence (6 points)
------------------------------------------

(a)
This code does not diverge. According to the programming guide,
thread index is given by:
    threadIdx = threadIdx.x + blockSize.x * (threadIdx.y + blockSize.y * threadIdx.z)

Therefore, for thread belonging to the same warp, their computed
idx would be x, 32 + x, 64 + x, ... Thus, idx % 32 would be the
same within the warp. So, all threads in the same warp perform
the same instruction, and no divergence happens.

(b)
This code diverges. Same as the analysis above, for threads in
the same wap, their threadIdx.x would be x, x + 1, x + 2, x + 3.
Therefore, they would loop for different number of times.


Question 1.3: Coalesced Memory Access (9 points)
------------------------------------------------

(a)
Yes. Since each cache line has 128 byte length, it contains 32 float data.
And because each warp has 32 threads, it writes to 1 cache lines.

(b)
No. Each warp writes to 32 cache lines.

(c)
No. Each warp writes to 2 cache lines.


Question 1.4: Bank Conflicts and Instruction Dependencies (15 points)
---------------------------------------------------------------------

(a)
There is no bank conflicts. When considering bank conflicts, we only need
to consider one single instruction. For thread in the same warp, for the
first line of code in the loop, the computation process can be broken into
three steps. First, data is loaded from lhs and rhs, then they gets multiplied,
and finally we store the result back to output. When loading data from lhs,
all threads access 32 different banks, so there is no bank conflict. When loading
from rhs, all threads access one same address, resulting in a broadcast, and
therefore no bank conflicts. When storing the result back to output, all threads
access 32 different banks and still no bank conflicts. This is the same to
the second line of code.

(b)
1. x = lhs[i + 32 * k];
2. y = rhs[k + 128 * j];
3. z = output[i + 32 * j];
4. z = z + x * y;
5. output[i + 32 * j] = z;

6. x = lhs[i + 32 * (k + 1)];
7. y = rhs[(k + 1) + 128 * j];
8. z = output[i + 32 * j];
9. z = z + x * y;
10. output[i + 32 * j] = z;

(c)
Line 4 depends on Line 1, 2 and 3.
Line 5 depends on Line 4.
Line 8 depends on Line 5.
Line 9 depends on Line 6, 7 and 8.
Line 10 depends on Line 9.

(d)
int i = threadIdx.x;
int j = threadIdx.y;
for (int k = 0; k < 128; k += 2) {
    float k1 = lhs[i + 32 * k] * rhs[k + 128 * j];
    float k2 = lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
    output[i + 32 * j] += (k1 + k2);
}

(e)
We can use more threads to hide latency.
We can unroll the loop and increase instruction level parallelism.




PART 2 - Matrix transpose optimization (65 points)
--------------------------------------------------

Size 512 naive CPU: 1.014656 ms
Size 512 GPU memcpy: 1.153408 ms
Size 512 naive GPU: 0.023552 ms
Size 512 shmem GPU: 0.011264 ms
Size 512 optimal GPU: 0.023552 ms

Size 1024 naive CPU: 13.294848 ms
Size 1024 GPU memcpy: 0.651744 ms
Size 1024 naive GPU: 0.071680 ms
Size 1024 shmem GPU: 0.022560 ms
Size 1024 optimal GPU: 0.072704 ms

Size 2048 naive CPU: 83.428513 ms
Size 2048 GPU memcpy: 1.142400 ms
Size 2048 naive GPU: 0.242688 ms
Size 2048 shmem GPU: 0.077824 ms
Size 2048 optimal GPU: 0.243712 ms

Size 4096 naive CPU: 405.569702 ms
Size 4096 GPU memcpy: 1.659872 ms
Size 4096 naive GPU: 0.944128 ms
Size 4096 shmem GPU: 0.296960 ms
Size 4096 optimal GPU: 1.001472 ms


BONUS (+5 points, maximum set score is 100 even with bonus)
--------------------------------------------------------------------------------

1. 2 calls to vec_add involves 6 global memory access while the following code involves
only 4 global memory access.
2. The latter has much higher instruction-level parallelism since there is less dependency.