#!/bin/sh
current_path=$(pwd)
rm -r ./upi_bench_result
vtune -collect memory-access -r upi_bench_result python3 bench.py --matrix-size 5000 --iterations 10 --numa-node both --num-tasks 20 --noHT
vtune-gui  $current_path/upi_bench_result/upi_bench_result.vtune 
