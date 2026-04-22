#!/bin/bash
# Script to run benchmarks and output results in CSV format
./bin/benchmark -b tensor_core --csv results.csv "$@"
echo "Benchmark completed. Results saved to results.csv"
