#!/bin/bash
# Script to run benchmarks and output results in CSV format
./bin/benchmark -b cublas_tensor_core_bench -b cutlass_tensor_core_bench --csv results_3090ti_missing.csv "$@"
echo "Benchmark completed. Results saved to results.csv"
