# Summation of vectors of size $2^{20}$

[Github](https://github.com/ErenOzilgili/Cython_PerformanceCheck/tree/main/CallFrom_cu_py)

`neofetch`:

```structured text
            .-/+oossssoo+/-.               erenozilgili@comp 
        `:+ssssssssssssssssss+:`           ----------------- 
      -+ssssssssssssssssssyyssss+-         OS: Ubuntu 24.04 LTS x86_64 
    .ossssssssssssssssssdMMMNysssso.       Host: ABRA A5 V16.4 
   /ssssssssssshdmmNNmmyNMMMMhssssss/      Kernel: 6.8.0-39-generic 
  +ssssssssshmydMMMMMMMNddddyssssssss+     Uptime: 1 day, 11 hours, 14 mins 
 /sssssssshNMMMyhhyyyyhmNMMMNhssssssss/    Packages: 1839 (dpkg), 15 (snap) 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Shell: bash 5.2.21 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   Resolution: 2560x1440 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   DE: GNOME 46.0 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   WM: Mutter 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   WM Theme: Adwaita 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Theme: Yaru-purple-dark [GTK2/3] 
 /sssssssshNMMMyhhyyyyhdNMMMNhssssssss/    Icons: Yaru-purple-dark [GTK2/3] 
  +sssssssssdmydMMMMMMMMddddyssssssss+     Terminal: gnome-terminal 
   /ssssssssssshdmNNNNmyNMMMMhssssss/      CPU: Intel i5-9300H (8) @ 4.100GHz 
    .ossssssssssssssssssdMMMNysssso.       GPU: Intel CoffeeLake-H GT2 [UHD Graphics 630] 
      -+sssssssssssssssssyyyssss+-         GPU: NVIDIA GeForce GTX 1650 Mobile / Max-Q 
        `:+ssssssssssssssssss+:`           Memory: 6957MiB / 15700MiB 
            .-/+oossssoo+/-.
                                                                   
                                                                   


```

`lscpu`:

```structured text
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          39 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   8
  On-line CPU(s) list:    0-7
Vendor ID:                GenuineIntel
  Model name:             Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz
    CPU family:           6
    Model:                158
    Thread(s) per core:   2
    Core(s) per socket:   4
    Socket(s):            1
    Stepping:             10
    CPU(s) scaling MHz:   161%
    CPU max MHz:          2400.0000
    CPU min MHz:          800.0000
    BogoMIPS:             4800.00
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fx
                          sr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts re
                          p_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est 
                          tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes
                           xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb pti ssbd ibrs ibpb stibp tpr_s
                          hadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust sgx bmi1 avx2 smep bmi2 erms invpcid mpx
                           rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp
                           hwp_notify hwp_act_window hwp_epp vnmi sgx_lc md_clear flush_l1d arch_capabilities
Virtualization features:  
  Virtualization:         VT-x
Caches (sum of all):      
  L1d:                    128 KiB (4 instances)
  L1i:                    128 KiB (4 instances)
  L2:                     1 MiB (4 instances)
  L3:                     8 MiB (1 instance)
NUMA:                     
  NUMA node(s):           1
  NUMA node0 CPU(s):      0-7
```

`cpupower frequency-info`:

```structured text
analyzing CPU 0:
  driver: intel_pstate
  CPUs which run at the same hardware frequency: 0
  CPUs which need to have their frequency coordinated by software: 0
  maximum transition latency:  Cannot determine or is not supported.
  hardware limits: 800 MHz - 2.40 GHz
  available cpufreq governors: performance powersave
  current policy: frequency should be within 800 MHz and 4.10 GHz.
                  The governor "performance" may decide which speed to use
                  within this range.
  current CPU frequency: Unable to call hardware
  current CPU frequency: 4.00 GHz (asserted by call to kernel)
  boost state support:
    Supported: yes
    Active: yes
```

`nvidia-smi`:

```structured text
Wed Aug 21 02:34:19 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1650        Off | 00000000:01:00.0 Off |                  N/A |
| N/A   51C    P0              19W /  50W |      6MiB /  4096MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1519      G   /usr/lib/xorg/Xorg                            4MiB |
+---------------------------------------------------------------------------------------+
```

## CUDA Python Comparison

CUDA function `funcX()` is called from CUDA. File name for CUDA executable is `testSum`.

Python calls the function `cyFuncX()` located in Cython and which immediately calls `funcX()` in CUDA. File name for Python is `pySum.py`.

Timings are collected with

```bash
#!/bin/bash

for i in $(seq 1 5)
do
	./testSum
done

for i in $(seq 1 5)
do
	python3 ./pySum.py
done
```

Every row below represents a one program run with 100 iterations.

Timings (with `cudaEventRecord` on CUDA and timeit on Python):

| CUDA (seconds) (amortized) | Python (seconds) (amortized) | CUDA (seconds) (total) | Python (seconds) (total) |
| -------------------------- | ---------------------------- | ---------------------- | ------------------------ |
| 0.0411606                  | 0.04368801468997845          | 4.11606                | 4.3688014689978445       |
| 0.0431044                  | 0.04371115242000087          | 4.31044                | 4.371115242000087        |
| 0.0428197                  | 0.04336321128997952          | 4.28197                | 4.336321128997952        |
| 0.0433299                  | 0.04396165964004467          | 4.33299                | 4.396165964004467        |
| 0.0433867                  | 0.043706826190027644         | 4.33867                | 4.370682619002764        |

Timings (with chrono on CUDA and timeit on Python):

| CUDA (seconds) (amortized) | Python (seconds) (amortized) | CUDA (seconds) (total) | Python (seconds) (total) |
| -------------------------- | ---------------------------- | ---------------------- | ------------------------ |
| 0.04289                    | 0.043900672260060676         | 4.289                  | 4.390067226006067        |
| 0.04336                    | 0.044821317919995635         | 4.336                  | 4.482131791999564        |
| 0.04369                    | 0.046987130519992204         | 4.369                  | 4.698713051999221        |
| 0.04514                    | 0.04612855004990706          | 4.514                  | 4.612855004990706        |
| 0.0433                     | 0.044534294230106755         | 4.33                   | 4.453429423010675        |

`hyperfine --prepare './testSum --reset' './testSum'`:

```structured text
Benchmark 1: ./testSum
  Time (mean ± σ):      4.563 s ±  0.036 s    [User: 4.409 s, System: 0.151 s]
  Range (min … max):    4.514 s …  4.614 s    10 runs
```

`hyperfine 'python3 ./pySum.py'`:

```structured text
Benchmark 1: python3 ./pySum.py
  Time (mean ± σ):      4.580 s ±  0.038 s    [User: 4.420 s, System: 0.159 s]
  Range (min … max):    4.554 s …  4.681 s    10 runs
```

**Note:** We use `cudaDeviceReset()` (or the `--reset` argument above) before each test to prevent time discrepancies caused by cold cache. Otherwise the initial test takes significantly longer (~%33 / about 2 seconds increase in total time).

# Codes Used In Timings

## CUDA:

Kernel `vectorAdd`:

```c++
//CUDA Kernel
__global__ void vectorAdd(int* a,
						  int* b,
						  int* c, int N){
	//Assing the thread id
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	//If within boundaries, sum up
	if(tid < N){
		c[tid] = a[tid] + b[tid];
	}
}
```

Function `funcX()`:

```c++
void funcX(){
    //Size of the vector addition
    int N = 1 << 20;

    size_t bytes = N * sizeof(int);

    int* a;
    int* b;
    int* c;

    a = (int*)malloc(bytes);
    b = (int*)malloc(bytes);
    c = (int*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    //Allocate memory on device (GPU)
    int *d_a, *d_b, *d_c;
      
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    //Copy data from host to device (CPU -> GPU)
    //Memory location of d_a (pointer) copied bytes amount of data from a (pointer)
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    //Threads per thread block 
    int threadNum_perThreadBl = 1024;

    //Number of thread blocks
    int numThreadBlock = (N + threadNum_perThreadBl - 1) / threadNum_perThreadBl;

    //Launch the kernel
    vectorAdd<<<numThreadBlock, threadNum_perThreadBl>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    //Copy sum of vectors (c) to host
    cudaMemcpy(c , d_c, bytes, cudaMemcpyDeviceToHost);

    //Validate Result
    verify_result(a, b, c, N);

    //Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
}
```

### How we timed in CUDA

`main()`:

```cpp
int main(){
    cudaDeviceReset();

    //(1)
    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    */

    float total_duration = 0;
    float duration = 0;

    int repetition = 100;

    for(int i = 0; i < repetition; i++){
        //Start the clock
        //(1)
        //cudaEventRecord(start);
        //(2)
        auto start = std::chrono::high_resolution_clock::now();

        //funcX call
        funcX();

        // End time
        //(1)
        //cudaEventRecord(stop);
        //(2)
        auto end = std::chrono::high_resolution_clock::now();

        //Wait for the stop event to complete
        //(1)
        //cudaEventSynchronize(stop);

        //(2)
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        //Calculate the elapsed time in milliseconds
        //(1)
        //cudaEventElapsedTime(&duration, start, stop);

        total_duration += duration;
    }

    std::cout << "Calculations are correct" << std::endl;
    std::cout << "Time it took in seconds (miliseconds * 10^3 =  seconds): " << (total_duration/1000) << "\n" << std::endl;
    std::cout << "Amortized time in seconds (miliseconds * 10^3 =  seconds): " << (total_duration/1000) / repetition << "\n" << std::endl;
}
```

## Cython

```cython
cdef extern from "matrixSum.h":
    void funcX()
    void resetDevice()

def cyFuncX():
    funcX()

def resetDev():
    resetDevice()
```

### How we timed in Python

```python
import timeit #Used to measure the time of the function executions
import wrappedCuda #Cuda

number = 100

prep="""
import wrappedCuda
wrappedCuda.resetDev()
"""

total = timeit.timeit(stmt="wrappedCuda.cyFuncX()",
                        setup=prep,
						number=100)

print("Repeating the function {} time, below are the execution times".format(number))
print("Time it took for -cuda wrapped with cython- is (sec): {}".format(total))
print("Time it took for -cuda wrapped with cython- is (sec) (amortized): {}\n".format(total/100))
```

