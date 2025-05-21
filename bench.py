# bench.py
#
# Cartouche :
# Objectif : Ce script effectue un tir de performance pour saturer le bus UPI (Ultra Path Interconnect)
#           lors de calculs inter-NUMA sur un systeme multi-noeuds NUMA. Il utilise des multiplications
#           de matrices parallellisees, avec des matrices d'entree allouees sur des noeuds NUMA specifiques
#           et des matrices resultats allouees sur le noeud oppose a l'execution, afin de maximiser les
#           acces memoire distants et le trafic UPI. Les performances sont analysees via des outils comme
#           Intel VTune Profiler.
# Contexte : Le script est concu pour tester les performances NUMA en controlant l'affinite CPU et memoire,
#            en repartissant les taches entre deux noeuds NUMA, et en generant une charge elevee via des
#            multiplications de matrices (via numpy.dot). Les resultats sont journalises pour analyse.
# Parametres configurables :
#   - --matrix-size : Taille des matrices carrees.
#   - --iterations : Nombre de multiplications par tache.
#   - --numa-node : Noeud(s) d'execution ('0', '1', ou 'both').
#   - --memory-node : Noeud pour les matrices d'entree.
#   - --num-tasks : Nombre de taches paralleles (2 a 40).
#   - --HT/--noHT : Utilisation ou non de l'hyper-threading.
# Dependances :
#   - Bibliotheques Python : psutil, numpy, threadpoolctl, pyyaml, tqdm (installer via pip).
#   - Bibliotheque systeme : libnuma.so pour la gestion NUMA.
# Auteurs : GDI
# Date : Mai 2025 (derniere mise a jour).
#

# Importation des bibliotheques necessaires
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

# Configuration des variables d'environnement pour limiter les threads BLAS a 1,
# evitant les interferences dans un contexte multi-processus.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Chargement et configuration de la bibliotheque libnuma pour la gestion NUMA
try:
    libnuma = ctypes.CDLL("libnuma.so")
    libnuma.numa_available.restype = ctypes.c_int
    if libnuma.numa_available() < 0:
        raise RuntimeError("NUMA n'est pas disponible sur ce systeme.")

    # Definition des signatures des fonctions libnuma pour l'allocation memoire,
    # l'affinite CPU/memoire, et la verification des noeuds.
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

# Classe Tee pour rediriger la sortie standard vers la console et un fichier journal
class Tee:
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout

    def write(self, message):
        # Ecrit le message a la fois dans la console et dans le fichier journal
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        # Synchronise les buffers de la console et du fichier
        self.terminal.flush()
        self.log_file.flush()

# Fonction pour obtenir la liste des CPU d'un noeud NUMA
def get_numa_node_cpus(node, use_ht=False):
    # Retourne 20 CPU par noeud sans hyper-threading (coeurs physiques) ou 40 avec
    if use_ht:
        return list(range(node * 40, (node + 1) * 40))
    else:
        return list(range(node * 20, (node + 1) * 20))

# Fonction pour determiner le noeud NUMA d'un CPU donne
def get_node_from_cpu(cpu_id, use_ht=False):
    # Mappe le CPU au noeud 0 ou 1 en fonction de l'hyper-threading
    if use_ht:
        return 0 if cpu_id < 40 else 1
    else:
        return 0 if cpu_id < 20 else 1

# Fonction pour formater une liste de CPU en une chaine lisible
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

# Fonction pour initialiser une matrice sur un noeud NUMA specifique
def initialize_matrix_on_node(matrix_size, node, matrix_ptr, size_bytes, matrix_name):
    # Definit le noeud prefere pour l'allocation memoire si NUMA est disponible
    if numa_available:
        libnuma.numa_set_preferred(node)
        print(f"Initialisation de la matrice {matrix_name} sur le noeud {node}.")

    # Remplit la matrice avec des valeurs aleatoires reproductibles
    random.seed(42)
    initial_data = [random.random() for _ in range(matrix_size * matrix_size)]
    ctypes.memmove(matrix_ptr, (ctypes.c_double * len(initial_data)).from_buffer(np.array(initial_data, dtype=np.float64)), size_bytes)

# Fonction executee par chaque tache parallele pour effectuer les multiplications de matrices
def matrix_multiply_task(task_id, matrix_size, iterations, cpu_id, num_tasks, matrix_a_ptr, matrix_b_ptr, results_queue, use_ht):
    try:
        # Limite les operations BLAS a un seul thread pour eviter les conflits
        with ThreadpoolController().limit(limits=1, user_api='blas'):
            # Definit et verifie l'affinite CPU pour le processus
            process = psutil.Process()
            process.cpu_affinity([cpu_id])
            actual_affinity = process.cpu_affinity()
            if cpu_id not in actual_affinity:
                raise RuntimeError(f"Tache {task_id} : Affinite incorrecte, attendue {cpu_id}, reelle {actual_affinity}")

            # Verifie que le processus utilise un seul thread
            num_threads = len(process.threads())
            if num_threads > 1:
                print(f"Avertissement Tache {task_id} : Trop de threads detectes ({num_threads}, attendu 1)")

            # Configure l'affinite memoire et le noeud prefere pour l'execution
            if numa_available:
                preferred_node = get_node_from_cpu(cpu_id, use_ht)
                nodemask = libnuma.numa_bitmask_alloc(2)
                try:
                    libnuma.numa_bitmask_setbit(nodemask, preferred_node)
                    libnuma.numa_set_membind(nodemask)
                    libnuma.numa_set_preferred(preferred_node)
                finally:
                    libnuma.numa_bitmask_free(nodemask)

            # Calcule la taille de la matrice et le noeud cible pour matrix_c
            size_bytes = matrix_size * matrix_size * ctypes.sizeof(ctypes.c_double)
            exec_node = get_node_from_cpu(cpu_id, use_ht)
            result_node = 1 if exec_node == 0 else 0  # Alloue matrix_c sur le noeud oppose

            # Force l'affinite memoire pour matrix_c sur le noeud cible
            if numa_available:
                nodemask = libnuma.numa_bitmask_alloc(2)
                try:
                    libnuma.numa_bitmask_setbit(nodemask, result_node)
                    libnuma.numa_set_membind(nodemask)
                    print(f"Tache {task_id} : Definition de l'affinite memoire sur le noeud {result_node} pour matrix_c.")
                finally:
                    libnuma.numa_bitmask_free(nodemask)

            # Alloue matrix_c avec une marge pour l'alignement sur 64 octets
            matrix_c_ptr = libnuma.numa_alloc_onnode(size_bytes + 64, result_node)
            if not matrix_c_ptr:
                raise MemoryError(f"Tache {task_id} : Echec de l'allocation memoire pour matrix_c sur le noeud {result_node}.")
            aligned_c_ptr = (matrix_c_ptr + 63) & ~63  # Alignement memoire

            # Verifie le noeud d'allocation reel de matrix_c
            if numa_available:
                membind = libnuma.numa_get_membind()
                node_0_set = libnuma.numa_bitmask_isbitset(membind, 0)
                node_1_set = libnuma.numa_bitmask_isbitset(membind, 1)
                allocated_node = 1 if node_1_set and not node_0_set else 0 if node_0_set and not node_1_set else -1
                print(f"Tache {task_id} : Matrice C allouee sur noeud {allocated_node} (attendu {result_node})")
                if allocated_node != result_node:
                    print(f"Tache {task_id} : Erreur d'allocation, noeud {allocated_node} ne correspond pas au noeud attendu {result_node}.")
                libnuma.numa_bitmask_free(membind)

            # Cree des vues numpy des matrices d'entree pour les calculs
            matrix_a = np.frombuffer(ctypes.string_at(matrix_a_ptr, size_bytes), dtype=np.float64).reshape(matrix_size, matrix_size)
            matrix_b = np.frombuffer(ctypes.string_at(matrix_b_ptr, size_bytes), dtype=np.float64).reshape(matrix_size, matrix_size)

            # Effectue les multiplications de matrices pour le nombre d'iterations specifie
            iteration_results = []
            block_size = matrix_size // 8  # Divise en blocs 8x8 pour augmenter le trafic UPI
            for _ in range(iterations):
                result = np.dot(matrix_a, matrix_b)
                # Copie les blocs de resultat dans matrix_c pour simuler des ecritures memoire
                for i in range(0, matrix_size, block_size):
                    for j in range(0, matrix_size, block_size):
                        block = result[i:i+block_size, j:j+block_size]
                        offset = (i * matrix_size + j) * ctypes.sizeof(ctypes.c_double)
                        ctypes.memmove(aligned_c_ptr + offset, block.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), block.nbytes)
                matrix_c = np.frombuffer(ctypes.string_at(aligned_c_ptr, size_bytes), dtype=np.float64).reshape(matrix_size, matrix_size)
                iteration_results.append(float(matrix_c[0, 0]))

            # Libere la memoire allouee pour matrix_c
            libnuma.numa_free(matrix_c_ptr, size_bytes + 64)

            # Envoie les resultats (metadonnees et premieres valeurs) a la file
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
        # Valide le nombre de taches
        if num_tasks < 2 or num_tasks > 40:
            raise ValueError("Erreur : Le nombre de taches doit etre entre 2 et 40")

        # Valide le noeud NUMA specifie
        valid_nodes = ['0', '1', 'both']
        if numa_node not in valid_nodes:
            raise ValueError(f"Erreur : Noeud NUMA {numa_node} invalide, valeurs attendues : {valid_nodes}")

        # Assigne les CPU en fonction du noeud NUMA et de l'hyper-threading
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

        # Verifie que suffisamment de CPU sont disponibles
        if num_tasks > len(cpu_ids):
            raise ValueError(f"Erreur : Pas assez de vCPUs ({len(cpu_ids)}) pour {num_tasks} taches")

        # Calcule la taille des matrices en octets
        size_bytes = matrix_size * matrix_size * ctypes.sizeof(ctypes.c_double)

        # Alloue les matrices d'entree sur les noeuds specifies
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

        # Verifie les allocations des matrices d'entree
        if numa_node == 'both':
            if not matrix_a0_ptr or not matrix_b0_ptr or not matrix_a1_ptr or not matrix_b1_ptr:
                raise MemoryError("Echec de l'allocation memoire avec numa_alloc_onnode.")
        else:
            if not matrix_a_ptr or not matrix_b_ptr:
                raise MemoryError("Echec de l'allocation memoire avec numa_alloc_onnode.")

        # Initialise les matrices d'entree avec des valeurs aleatoires
        if numa_node == 'both':
            initialize_matrix_on_node(matrix_size, 0, matrix_a0_ptr, size_bytes, "A0")
            initialize_matrix_on_node(matrix_size, 0, matrix_b0_ptr, size_bytes, "B0")
            initialize_matrix_on_node(matrix_size, 1, matrix_a1_ptr, size_bytes, "A1")
            initialize_matrix_on_node(matrix_size, 1, matrix_b1_ptr, size_bytes, "B1")
        else:
            initialize_matrix_on_node(matrix_size, memory_node, matrix_a_ptr, size_bytes, "A")
            initialize_matrix_on_node(matrix_size, memory_node, matrix_b_ptr, size_bytes, "B")

        # Affiche les noeuds cibles pour matrix_c
        if numa_node == 'both':
            for i in range(num_tasks):
                exec_node = get_node_from_cpu(cpu_ids[i], use_ht)
                result_node = 1 if exec_node == 0 else 0
                print(f"Matrice C pour la tache {i} allouee sur le noeud {result_node} (execution sur noeud {exec_node}).")
        else:
            for i in range(num_tasks):
                print(f"Matrice C pour la tache {i} allouee sur le noeud {1 if memory_node == 0 else 0} (execution sur noeud {memory_node}).")

        # Cree une file pour collecter les resultats des taches
        results_queue = multiprocessing.Queue()
        processes = []

        # Lance les processus pour chaque tache
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

        # Attend la fin de tous les processus avec une barre de progression
        print("Attente de la fin des processus...")
        for p in tqdm(processes, desc="Progression des taches", unit="tache"):
            p.join()

        # Collecte les resultats des taches
        results = [None] * num_tasks
        for _ in range(num_tasks):
            result = results_queue.get(timeout=10)
            if 'error' in result:
                print(f"Tache {result['task_id']} a echoue : {result['error']}")
                continue
            results[result['task_id']] = result

        # Affiche les details de chaque tache
        for result in results:
            if result is None:
                continue
            print(f"Tache {result['task_id']} : vCPU {result['cpu_id']}, "
                  f"affinite reelle : {result['affinity']}, "
                  f"threads : {result['threads']}, "
                  f"noeud NUMA exec : {result['exec_node']}, "
                  f"noeud memoire resultat : {result.get('result_node', memory_node_display)}")

        # Affiche les statistiques NUMA pour le processus principal
        current_pid = os.getpid()
        numastat_output = subprocess.run(['numastat', '-p', str(current_pid)], capture_output=True, text=True)
        if numastat_output.returncode == 0:
            print(f"\nStatistiques NUMA pour PID {current_pid} :")
            print(numastat_output.stdout)

        # Libere la memoire allouee pour les matrices d'entree
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

# Fonction principale pour parser les arguments et lancer l'execution
def main():
    # Cree un fichier journal horodate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"execution_log_{timestamp}.log"
    with open(log_filename, 'w') as log_file:
        sys.stdout = Tee(log_file)
        try:
            # Parse les arguments de la ligne de commande
            parser = argparse.ArgumentParser()
            parser.add_argument("--matrix-size", type=int, default=100)
            parser.add_argument("--iterations", type=int, default=10)
            parser.add_argument("--numa-node", default="0")
            parser.add_argument("--memory-node", type=int, default=None)
            parser.add_argument("--num-tasks", type=int, default=2)
            parser.add_argument("--HT", action="store_true")
            parser.add_argument("--noHT", action="store_true")
            args = parser.parse_args()

            # Determine l'utilisation de l'hyper-threading et le noeud memoire
            use_ht = args.HT and not args.noHT
            memory_node = args.memory_node if args.memory_node is not None else int(args.numa_node) if args.numa_node != 'both' else 0

            # Configure l'affinite memoire globale si necessaire
            if numa_available and args.numa_node != 'both':
                nodemask = libnuma.numa_bitmask_alloc(2)
                libnuma.numa_bitmask_setbit(nodemask, memory_node)
                libnuma.numa_set_membind(nodemask)
                libnuma.numa_bitmask_free(nodemask)

            # Lance l'execution des taches
            matrix_multiply(args.matrix_size, args.iterations, args.numa_node, memory_node, args.num_tasks, use_ht)

        finally:
            sys.stdout = sys.__stdout__
            print(f"Journalisation terminee. Log disponible dans {log_filename}")

if __name__ == "__main__":
    main()
