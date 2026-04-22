#!/bin/bash
# Script to run benchmarks and output results in CSV format
./bin/benchmark -b shared -b shared_32 --csv results.csv "$@"
echo "Benchmark completed. Results saved to results.csv"
