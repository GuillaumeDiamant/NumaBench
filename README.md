Matrix Multiplication NUMA Benchmark
Project Overview

The bench.py project is a performance benchmarking tool designed to stress the Ultra Path Interconnect (UPI) bus in a multi-node Non-Uniform Memory Access (NUMA) system. It achieves this by performing parallel matrix non optimal multiplications with memory allocations strategically placed across NUMA nodes to maximize remote memory accesses, thereby saturating the UPI bus. The tool is intended for performance analysis using profiling tools like Intel VTune Profiler, focusing on memory access patterns and UPI traffic.
The script executes matrix multiplications in multiple processes, each pinned to a specific CPU, with input matrices allocated on designated NUMA nodes and result matrices allocated on the opposite node to the execution. This setup ensures high inter-node memory traffic, making it ideal for studying NUMA performance and UPI bottlenecks.
Methodology
The benchmarking methodology involves the following steps:

Matrix Allocation: Allocates input matrices (A and B) on specified NUMA nodes (0, 1, or both) using libnuma. The result matrix (C) for each task is allocated on the NUMA node opposite to the execution node (e.g., node 1 for tasks on node 0, and vice versa) to force remote memory accesses.
Parallel Execution: Launches multiple processes (2 to 40 tasks), each performing a specified number of matrix multiplications (numpy.dot) on matrices of configurable size. Each process is bound to a specific CPU using psutil to control affinity.
UPI Stress: Divides the result matrix into 8x8 blocks to increase memory transactions, amplifying UPI traffic due to frequent remote writes.
Thread Control: Ensures single-threaded BLAS operations using threadpoolctl and environment variables (OPENBLAS_NUM_THREADS=1, etc.) to avoid thread contention.
Logging and Profiling: Logs execution details (CPU affinity, NUMA node allocation, thread count) to a timestamped file. Collects NUMA statistics via numastat and supports profiling with VTune for detailed memory and UPI analysis.

Components
The bench.py script consists of the following key components:

NUMA Integration: Uses libnuma.so via ctypes for memory allocation (numa_alloc_onnode), CPU-to-node mapping (numa_node_of_cpu), and memory affinity (numa_set_membind, numa_set_preferred).
Parallel Processing: Employs multiprocessing to run tasks in parallel, with each process executing matrix_multiply_task for matrix operations.
Matrix Operations: Leverages numpy for efficient matrix multiplications and memory management.
CPU Affinity: Uses psutil to pin processes to specific CPUs and verify affinity.
Thread Management: Utilizes threadpoolctl to limit BLAS operations to a single thread per process.
Logging: Implements a Tee class to redirect console output to a timestamped log file (e.g., execution_log_20250520_181235.log).
Progress Tracking: Uses tqdm to display a progress bar for task completion.
Command-Line Interface: Parses arguments with argparse for configurable parameters (matrix size, iterations, NUMA node, etc.).

Prerequisites
To run the bench.py script, ensure the following requirements are met:
Software Requirements

Python 3.6+: Required for script execution.
Python Libraries:
psutil: For CPU affinity management.
numpy: For matrix operations.
threadpoolctl: For controlling BLAS threads.
pyyaml: For configuration (not used in current version but listed as dependency).
tqdm: For progress bar display.
Install via: pip install --upgrade psutil numpy threadpoolctl pyyaml tqdm


libnuma: System library for NUMA operations. Install on Debian/Ubuntu with:sudo apt-get install libnuma-dev


numactl: Command-line tool for NUMA statistics and system verification. Install with:sudo apt-get install numactl


Intel VTune Profiler (optional): For memory access and UPI traffic analysis. Ensure it is installed and configured for memory-access collection.

Hardware Requirements

NUMA System: A multi-node NUMA system with at least two nodes (e.g., dual-socket Intel Xeon server). Each node should have sufficient CPUs (minimum 20 physical cores per node for default settings) and memory (~100 MB free per node for 500x500 matrices).
CPU Support: Supports hyper-threading (optional, controlled via --HT/--noHT flags). Without hyper-threading, 20 CPUs per node are assumed.
Memory: Sufficient free memory per NUMA node (e.g., 60 GB free, as verified by numactl --hardware).

System Verification
Before running, verify the NUMA configuration with:
numactl --hardware

Expected output should show at least two nodes with available CPUs and memory, e.g.:
available: 2 nodes (0-1)
node 0 cpus: 0-19
node 0 size: 64019 MB
node 0 free: 60257 MB
node 1 cpus: 20-39
node 1 size: 64501 MB
node 1 free: 44066 MB

Usage
To execute the benchmark, run the script with desired parameters. Example command using VTune for profiling:
vtune -collect memory-access -r upi_bench_result python3 bench.py --matrix-size 500 --iterations 100 --numa-node both --num-tasks 20 --noHT

Command-Line Arguments

--matrix-size <int>: Size of square matrices (default: 100).
--iterations <int>: Number of matrix multiplications per task (default: 10).
--numa-node <str>: NUMA node(s) for execution (0, 1, or both; default: 0).
--memory-node <int>: NUMA node for input matrices (default: same as --numa-node or 0 for both).
--num-tasks <int>: Number of parallel tasks (2 to 40; default: 2).
--HT/--noHT: Enable or disable hyper-threading (default: disabled).

Output
The script generates the following outputs:

Log File: A timestamped log file (e.g., execution_log_20250520_181235.log) containing:
Matrix initialization details (e.g., "Initialization of matrix A0 on node 0").
Task allocation details (e.g., "Matrix C for task 0 allocated on node 1 (execution on node 0)").
Task execution summary (CPU, affinity, threads, NUMA nodes).
NUMA statistics from numastat.


VTune Results (if used): A directory (e.g., upi_bench_result) with profiling data on memory accesses, UPI traffic, and CPU utilization.
Console Output: Mirrors the log file content, with a progress bar for task completion.

License
This project is licensed under the MIT License. See the LICENSE file for details (not included in this repository).
