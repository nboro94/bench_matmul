#!/usr/bin/env python3

import subprocess
import os
import time
import argparse
from datetime import datetime

def get_available_log_dir():
    """Find an available log directory."""
    base_dir = "logs"
    
    # If logs doesn't exist, use it
    if not os.path.exists(base_dir):
        return base_dir
    
    # If logs exists, try logs_1, logs_2, etc.
    counter = 1
    while True:
        new_dir = f"{base_dir}_{counter}"
        if not os.path.exists(new_dir):
            return new_dir
        counter += 1

def run_benchmark(executable_path, matrix_size, log_dir):
    """Run the matrix multiplication benchmark with the specified size."""
    print(f"Running benchmark with matrix size {matrix_size}x{matrix_size}...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/matmul_bench_{matrix_size}_{timestamp}.log"
    
    start_time = time.time()
    
    try:
        # Run the benchmark and capture output
        with open(log_file, 'w') as f:
            f.write(f"Matrix Multiplication Benchmark - Size: {matrix_size}x{matrix_size}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Execute the benchmark and redirect output to the log file
            process = subprocess.run([executable_path, str(matrix_size)],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    universal_newlines=True,
                                    check=True)
            
            f.write(process.stdout)
        
        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed:.2f} seconds. Log saved to: {log_file}")
        return True
    
    except subprocess.CalledProcessError as e:
        with open(log_file, 'a') as f:
            f.write(f"\nERROR: Process failed with exit code {e.returncode}\n")
            if e.stdout:
                f.write(e.stdout)
        
        print(f"✗ Failed with exit code {e.returncode}. Log saved to: {log_file}")
        return False
    
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"\nERROR: {str(e)}\n")
        
        print(f"✗ Exception occurred: {str(e)}. Log saved to: {log_file}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run matrix multiplication benchmarks with various sizes')
    parser.add_argument('--executable', default='./bench-matmul', 
                        help='Path to the benchmark executable (default: ./bench-matmul)')
    args = parser.parse_args()
    
    # Matrix sizes to benchmark
    matrix_sizes = [64, 128, 512, 1024]
    
    # Get a single log directory for this run
    log_dir = get_available_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    print(f"Starting benchmark suite with {len(matrix_sizes)} different matrix sizes")
    print(f"Executable: {args.executable}")
    print(f"Logs will be saved to: {log_dir}")
    
    start_time = time.time()
    successful_runs = 0
    
    for size in matrix_sizes:
        if run_benchmark(args.executable, size, log_dir):
            successful_runs += 1
        else:
            assert(0)
        print("-" * 50)
    
    total_time = time.time() - start_time
    print(f"Benchmark suite completed: {successful_runs}/{len(matrix_sizes)} successful runs")
    print(f"Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
