# bench.py
#
# Objective: This script conducts a performance test to saturate the UPI (Ultra Path Interconnect) bus during inter-NUMA computations on a multi-node NUMA system. It employs parallelized matrix multiplications, with input matrices allocated on specific NUMA nodes and result matrices allocated on the node opposite to the execution, to maximize remote memory accesses and UPI traffic. Performance is analyzed using tools such as Intel VTune Profiler.
# Context: The script is designed to test NUMA performance by controlling CPU and memory affinity, distributing tasks across two NUMA nodes, and generating high computational load through matrix multiplications (using numpy.dot). Results are logged for analysis.
# Platform : Linux 
# Configurable Parameters:
# --matrix-size: Size of the square matrices.
# --iterations: Number of multiplications per task.
# --numa-node: Execution node(s) ('0', '1', or 'both').
# --memory-node: Node for input matrices.
# --num-tasks: Number of parallel tasks (2 to 40).
# --HT/--noHT: Enable or disable hyper-threading.
# Dependencies:
# Python libraries: psutil, numpy, threadpoolctl, pyyaml, tqdm (install via pip).
# System library: libnuma.so for NUMA management.
# Authors: Guillaume Diamant
# Date: May 2025 (last updated).
#

# Importing the necessary libraries
import sys
import time
import random
import multiprocessing
import argparse
import psutil
import os
import numpy as np
from datetime import datetime
from threadpoolctl import ThreadpoolController
import ctypes
import subprocess
from tqdm import tqdm

# Configure environment variables to limit BLAS threads to 1,
# avoiding interference in a multi-process context.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Loading and configuring the libnuma library for NUMA management
try:
    libnuma = ctypes.CDLL("libnuma.so")
    libnuma.numa_available.restype = ctypes.c_int
    if libnuma.numa_available() < 0:
        raise RuntimeError("NUMA n'est pas disponible sur ce systeme.")

# Define libnuma function signatures for memory allocation,
# CPU/memory affinity, and node verification.
    libnuma.numa_bitmask_alloc.restype = ctypes.c_void_p
    libnuma.numa_bitmask_alloc.argtypes = [ctypes.c_uint]
    libnuma.numa_bitmask_setbit.restype = None
    libnuma.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    libnuma.numa_set_membind.restype = None
    libnuma.numa_set_membind.argtypes = [ctypes.c_void_p]
    libnuma.numa_bitmask_free.restype = None
    libnuma.numa_bitmask_free.argtypes = [ctypes.c_void_p]
    libnuma.numa_alloc_onnode.restype = ctypes.c_void_p
    libnuma.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
    libnuma.numa_free.restype = None
    libnuma.numa_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libnuma.numa_set_preferred.restype = None
    libnuma.numa_set_preferred.argtypes = [ctypes.c_int]
    libnuma.numa_node_of_cpu.restype = ctypes.c_int
    libnuma.numa_node_of_cpu.argtypes = [ctypes.c_int]
    libnuma.numa_get_membind.restype = ctypes.c_void_p
    libnuma.numa_bitmask_isbitset.restype = ctypes.c_int
    libnuma.numa_bitmask_isbitset.argtypes = [ctypes.c_void_p, ctypes.c_uint]

    numa_available = True
    print("libnuma charge avec succes via ctypes.")
except OSError as e:
    numa_available = False
    print(f"Erreur lors du chargement de libnuma : {e}")
except RuntimeError as e:
    numa_available = False
    print(str(e))

# Tee class to redirect standard output to the console and a log file
class Tee:
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout

    def write(self, message):
       # Writes the message to both the console and the log file
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
   # Synchronizes console and file buffers
        self.terminal.flush()
        self.log_file.flush()

# Function to get the list of CPUs of a NUMA node
def get_numa_node_cpus(node, use_ht=False):
    # Returns 20 VCPUs per node without hyperthreading (physical cores) or 40 with
    if use_ht:
        return list(range(node * 40, (node + 1) * 40))
    else:
        return list(range(node * 20, (node + 1) * 20))

# Function to determine the NUMA node of a given CPU
def get_node_from_cpu(cpu_id, use_ht=False):
  # Maps CPU to node 0 or 1 depending on hyper-threading
    if use_ht:
        return 0 if cpu_id < 40 else 1
    else:
        return 0 if cpu_id < 20 else 1

# Function to format a list of CPUs into a readable string
def format_cpu_range(cpu_ids):
    if not cpu_ids:
        return "Aucun CPU"
    if len(cpu_ids) == 1:
        return str(cpu_ids[0])
    min_cpu = min(cpu_ids)
    max_cpu = max(cpu_ids)
    if all(i in cpu_ids for i in range(min_cpu, max_cpu + 1)):
        return f"{min_cpu}-{max_cpu}"
    return ",".join(str(i) for i in sorted(cpu_ids))

# Function to initialize a matrix on a specific NUMA node
def initialize_matrix_on_node(matrix_size, node, matrix_ptr, size_bytes, matrix_name):
    # Sets the preferred node for memory allocation if NUMA is available
    if numa_available:
        libnuma.numa_set_preferred(node)
        print(f"Initialisation de la matrice {matrix_name} sur le noeud {node}.")

   # Fill the matrix with random values
    random.seed(42)
    initial_data = [random.random() for _ in range(matrix_size * matrix_size)]
    ctypes.memmove(matrix_ptr, (ctypes.c_double * len(initial_data)).from_buffer(np.array(initial_data, dtype=np.float64)), size_bytes)

# Function executed by each parallel task to perform matrix multiplications
def matrix_multiply_task(task_id, matrix_size, iterations, cpu_id, num_tasks, matrix_a_ptr, matrix_b_ptr, results_queue, use_ht):
    try:
       # Limit BLAS operations to a single thread to avoid conflicts
        with ThreadpoolController().limit(limits=1, user_api='blas'):
 # Define and check the CPU affinity for the process
            process = psutil.Process()
            process.cpu_affinity([cpu_id])
            actual_affinity = process.cpu_affinity()
            if cpu_id not in actual_affinity:
                raise RuntimeError(f"Tache {task_id} : Affinite incorrecte, attendue {cpu_id}, reelle {actual_affinity}")

           # Checks that the process uses a single thread
            num_threads = len(process.threads())
            if num_threads > 1:
                print(f"Avertissement Tache {task_id} : Trop de threads detectes ({num_threads}, attendu 1)")

            # Configure memory affinity and preferred node for execution
            if numa_available:
                preferred_node = get_node_from_cpu(cpu_id, use_ht)
                nodemask = libnuma.numa_bitmask_alloc(2)
                try:
                    libnuma.numa_bitmask_setbit(nodemask, preferred_node)
                    libnuma.numa_set_membind(nodemask)
                    libnuma.numa_set_preferred(preferred_node)
                finally:
                    libnuma.numa_bitmask_free(nodemask)

            # Calculates the matrix size and target node for matrix_c
            size_bytes = matrix_size * matrix_size * ctypes.sizeof(ctypes.c_double)
            exec_node = get_node_from_cpu(cpu_id, use_ht)
            result_node = 1 if exec_node == 0 else 0  # Allocate matrix_c on the opposite node

            # Force memory affinity for matrix_c on the target node
            if numa_available:
                nodemask = libnuma.numa_bitmask_alloc(2)
                try:
                    libnuma.numa_bitmask_setbit(nodemask, result_node)
                    libnuma.numa_set_membind(nodemask)
                    print(f"Tache {task_id} : Definition de l'affinite memoire sur le noeud {result_node} pour matrix_c.")
                finally:
                    libnuma.numa_bitmask_free(nodemask)

           # Allocates matrix_c with a margin for 64-byte alignment
            matrix_c_ptr = libnuma.numa_alloc_onnode(size_bytes + 64, result_node)
            if not matrix_c_ptr:
                raise MemoryError(f"Tache {task_id} : Echec de l'allocation memoire pour matrix_c sur le noeud {result_node}.")
            aligned_c_ptr = (matrix_c_ptr + 63) & ~63  # Memory alignment

            # Check the actual allocation node of matrix_c
            if numa_available:
                membind = libnuma.numa_get_membind()
                node_0_set = libnuma.numa_bitmask_isbitset(membind, 0)
                node_1_set = libnuma.numa_bitmask_isbitset(membind, 1)
                allocated_node = 1 if node_1_set and not node_0_set else 0 if node_0_set and not node_1_set else -1
                print(f"Tache {task_id} : Matrice C allouee sur noeud {allocated_node} (attendu {result_node})")
                if allocated_node != result_node:
                    print(f"Tache {task_id} : Erreur d'allocation, noeud {allocated_node} ne correspond pas au noeud attendu {result_node}.")
                libnuma.numa_bitmask_free(membind)

           # Create numpy views of input matrices for computations
            matrix_a = np.frombuffer(ctypes.string_at(matrix_a_ptr, size_bytes), dtype=np.float64).reshape(matrix_size, matrix_size)
            matrix_b = np.frombuffer(ctypes.string_at(matrix_b_ptr, size_bytes), dtype=np.float64).reshape(matrix_size, matrix_size)

          # Performs matrix multiplications for the specified number of iterations
            iteration_results = []
            block_size = matrix_size // 8  # Split into 8x8 blocks to increase UPI traffic
            for _ in range(iterations):
                result = np.dot(matrix_a, matrix_b)
               # Copy result blocks into matrix c to simulate memory writes
                for i in range(0, matrix_size, block_size):
                    for j in range(0, matrix_size, block_size):
                        block = result[i:i+block_size, j:j+block_size]
                        offset = (i * matrix_size + j) * ctypes.sizeof(ctypes.c_double)
                        ctypes.memmove(aligned_c_ptr + offset, block.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), block.nbytes)
                matrix_c = np.frombuffer(ctypes.string_at(aligned_c_ptr, size_bytes), dtype=np.float64).reshape(matrix_size, matrix_size)
                iteration_results.append(float(matrix_c[0, 0]))

           # Free the memory allocated for matrix c
            libnuma.numa_free(matrix_c_ptr, size_bytes + 64)

            # Sends the results (metadata and first values) to the queue
            results_queue.put({
                'task_id': task_id,
                'cpu_id': cpu_id,
                'affinity': actual_affinity,
                'threads': num_threads,
                'iteration_results': iteration_results,
                'exec_node': exec_node,
                'result_node': result_node
            })
    except Exception as e:
        print(f"Tache {task_id} : Erreur rencontree : {str(e)}")
        results_queue.put({'task_id': task_id, 'error': str(e)})

# Fonction principale pour coordonner l'execution des taches NUMA
def matrix_multiply(matrix_size, iterations, numa_node, memory_node, num_tasks, use_ht=False):
    try:
       # Validates the number of spots
        if num_tasks < 2 or num_tasks > 40:
            raise ValueError("Erreur : Le nombre de taches doit etre entre 2 et 40")

       # Validates the specified NUMA node
        valid_nodes = ['0', '1', 'both']
        if numa_node not in valid_nodes:
            raise ValueError(f"Erreur : Noeud NUMA {numa_node} invalide, valeurs attendues : {valid_nodes}")

        # Assigns CPUs based on NUMA node and hyper-threading
        if numa_node == 'both':
            node0_cpus = get_numa_node_cpus(0, use_ht)
            node1_cpus = get_numa_node_cpus(1, use_ht)
            tasks_per_node = num_tasks // 2
            extra_tasks = num_tasks % 2
            cpu_ids = node0_cpus[:tasks_per_node + extra_tasks] + node1_cpus[:tasks_per_node]
            impacted_nodes = [0, 1]
        else:
            numa_node_int = int(numa_node)
            cpu_ids = get_numa_node_cpus(numa_node_int, use_ht)[:num_tasks]
            impacted_nodes = [numa_node_int]

        # Checks that enough CPUs are available
        if num_tasks > len(cpu_ids):
            raise ValueError(f"Erreur : Pas assez de vCPUs ({len(cpu_ids)}) pour {num_tasks} taches")

        # Calculates the size of matrices in bytes
        size_bytes = matrix_size * matrix_size * ctypes.sizeof(ctypes.c_double)

        # Allocates the input matrices on the specified nodes
        if numa_node == 'both':
            matrix_a0_ptr = libnuma.numa_alloc_onnode(size_bytes, 0)
            matrix_b0_ptr = libnuma.numa_alloc_onnode(size_bytes, 0)
            matrix_a1_ptr = libnuma.numa_alloc_onnode(size_bytes, 1)
            matrix_b1_ptr = libnuma.numa_alloc_onnode(size_bytes, 1)
            memory_node_display = "mixte"
        else:
            matrix_a_ptr = libnuma.numa_alloc_onnode(size_bytes, memory_node)
            matrix_b_ptr = libnuma.numa_alloc_onnode(size_bytes, memory_node)
            memory_node_display = str(memory_node)

       # Check input matrix allocations
        if numa_node == 'both':
            if not matrix_a0_ptr or not matrix_b0_ptr or not matrix_a1_ptr or not matrix_b1_ptr:
                raise MemoryError("Echec de l'allocation memoire avec numa_alloc_onnode.")
        else:
            if not matrix_a_ptr or not matrix_b_ptr:
                raise MemoryError("Echec de l'allocation memoire avec numa_alloc_onnode.")

        # Initializes the input matrices with random values
        if numa_node == 'both':
            initialize_matrix_on_node(matrix_size, 0, matrix_a0_ptr, size_bytes, "A0")
            initialize_matrix_on_node(matrix_size, 0, matrix_b0_ptr, size_bytes, "B0")
            initialize_matrix_on_node(matrix_size, 1, matrix_a1_ptr, size_bytes, "A1")
            initialize_matrix_on_node(matrix_size, 1, matrix_b1_ptr, size_bytes, "B1")
        else:
            initialize_matrix_on_node(matrix_size, memory_node, matrix_a_ptr, size_bytes, "A")
            initialize_matrix_on_node(matrix_size, memory_node, matrix_b_ptr, size_bytes, "B")

       # Displays target nodes for matrix_c
        if numa_node == 'both':
            for i in range(num_tasks):
                exec_node = get_node_from_cpu(cpu_ids[i], use_ht)
                result_node = 1 if exec_node == 0 else 0
                print(f"Matrice C pour la tache {i} allouee sur le noeud {result_node} (execution sur noeud {exec_node}).")
        else:
            for i in range(num_tasks):
                print(f"Matrice C pour la tache {i} allouee sur le noeud {1 if memory_node == 0 else 0} (execution sur noeud {memory_node}).")

        # Create a queue to collect task results
        results_queue = multiprocessing.Queue()
        processes = []

        # Launches processes for each task
        for i in range(num_tasks):
            if numa_node == 'both':
                exec_node = get_node_from_cpu(cpu_ids[i], use_ht)
                matrix_a_ptr = matrix_a0_ptr if exec_node == 0 else matrix_a1_ptr
                matrix_b_ptr = matrix_b0_ptr if exec_node == 0 else matrix_b1_ptr
            else:
                matrix_a_ptr = matrix_a_ptr
                matrix_b_ptr = matrix_b_ptr
            p = multiprocessing.Process(target=matrix_multiply_task, args=(i, matrix_size, iterations, cpu_ids[i], num_tasks, matrix_a_ptr, matrix_b_ptr, results_queue, use_ht))
            processes.append(p)
            p.start()

       # Waits for all processes to complete with a progress bar
        print("Attente de la fin des processus...")
        for p in tqdm(processes, desc="Progression des taches", unit="tache"):
            p.join()

       # Collects task results
        results = [None] * num_tasks
        for _ in range(num_tasks):
            result = results_queue.get(timeout=10)
            if 'error' in result:
                print(f"Tache {result['task_id']} a echoue : {result['error']}")
                continue
            results[result['task_id']] = result

        # Displays details of each task
        for result in results:
            if result is None:
                continue
            print(f"Tache {result['task_id']} : vCPU {result['cpu_id']}, "
                  f"affinite reelle : {result['affinity']}, "
                  f"threads : {result['threads']}, "
                  f"noeud NUMA exec : {result['exec_node']}, "
                  f"noeud memoire resultat : {result.get('result_node', memory_node_display)}")

       # Displays NUMA statistics for the main process
        current_pid = os.getpid()
        numastat_output = subprocess.run(['numastat', '-p', str(current_pid)], capture_output=True, text=True)
        if numastat_output.returncode == 0:
            print(f"\nStatistiques NUMA pour PID {current_pid} :")
            print(numastat_output.stdout)

       # Frees memory allocated for input matrices
        if numa_node == 'both':
            libnuma.numa_free(matrix_a0_ptr, size_bytes)
            libnuma.numa_free(matrix_b0_ptr, size_bytes)
            libnuma.numa_free(matrix_a1_ptr, size_bytes)
            libnuma.numa_free(matrix_b1_ptr, size_bytes)
        else:
            libnuma.numa_free(matrix_a_ptr, size_bytes)
            libnuma.numa_free(matrix_b_ptr, size_bytes)

    except Exception as e:
        print(str(e))
        sys.exit(1)

# Main function to parse arguments and start execution
def main():
    # Create a timestamp log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"execution_log_{timestamp}.log"
    with open(log_filename, 'w') as log_file:
        sys.stdout = Tee(log_file)
        try:
            # Parse command line arguments
            parser = argparse.ArgumentParser()
            parser.add_argument("--matrix-size", type=int, default=100)
            parser.add_argument("--iterations", type=int, default=10)
            parser.add_argument("--numa-node", default="0")
            parser.add_argument("--memory-node", type=int, default=None)
            parser.add_argument("--num-tasks", type=int, default=2)
            parser.add_argument("--HT", action="store_true")
            parser.add_argument("--noHT", action="store_true")
            args = parser.parse_args()

           # Determines hyper-threading usage and memory node
            use_ht = args.HT and not args.noHT
            memory_node = args.memory_node if args.memory_node is not None else int(args.numa_node) if args.numa_node != 'both' else 0

            # Configure global memory affinity if needed
            if numa_available and args.numa_node != 'both':
                nodemask = libnuma.numa_bitmask_alloc(2)
                libnuma.numa_bitmask_setbit(nodemask, memory_node)
                libnuma.numa_set_membind(nodemask)
                libnuma.numa_bitmask_free(nodemask)

            # Starts the execution of tasks
            matrix_multiply(args.matrix_size, args.iterations, args.numa_node, memory_node, args.num_tasks, use_ht)

        finally:
            sys.stdout = sys.__stdout__
            print(f"Journalisation terminee. Log disponible dans {log_filename}")

if __name__ == "__main__":
    main()
