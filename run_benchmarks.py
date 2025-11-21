#!/usr/bin/env python3

import subprocess
import os
import time
import argparse
from datetime import datetime
import concurrent.futures

# Short aliases mapping -> full method names in the C++ binary
ALIASES = {
    'naive': 'Naive-ijkLoop',
    'tiled': 'BlockTiled-CacheAware',
    'avx2': 'SIMD-AVX2-Transposed',
    'avx2direct': 'SIMD-AVX2-Direct',
    'transposed': 'RowColumn-Transposed',
    'scalar': 'Scalar-LoopUnrolled',
    'par-avx2': 'Parallel-SIMD-AVX2',
    'par-scalar': 'Parallel-Scalar-LoopUnrolled',
    'par-avx2-direct': 'Parallel-SIMD-Direct',
    'local': 'BlockLocal-StackTranspose'
}

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

def run_benchmark(executable_path, matrix_size, log_dir, run_methods=None):
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
            
            # Build command and execute the benchmark, redirecting output to the log file
            cmd = [executable_path, str(matrix_size)]
            if run_methods:
                # pass through the run methods as a single --run argument
                cmd.append(f"--run={run_methods}")

            process = subprocess.run(cmd,
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
    parser.add_argument('--run', default=None,
                        help='Comma-separated list of multiplication methods to run (passed through to the executable as --run=name1,name2). Aliases supported.')
    parser.add_argument('--sizes', default=None,
                        help='Optional comma-separated list of matrix sizes to run (overrides defaults). Example: --sizes=64,128')
    parser.add_argument('--list', action='store_true', help='Query the executable for available multiplication methods and exit')
    parser.add_argument('-j', '--jobs', type=int, default=1, help='Number of concurrent runs (default: 1)')
    args = parser.parse_args()

    # If user asked to list available methods, query the executable and exit early
    if args.list:
        try:
            res = subprocess.run([args.executable, '--list'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, check=True)
            print(res.stdout)
            return
        except subprocess.CalledProcessError as e:
            print('ERROR: failed to query executable with --list')
            print(e.output if hasattr(e, 'output') else str(e))
            return

    # Matrix sizes to benchmark (default set)
    default_sizes = [64, 128, 512, 1024]
    if args.sizes:
        try:
            matrix_sizes = [int(s) for s in args.sizes.split(',') if s.strip()]
            if not matrix_sizes:
                matrix_sizes = default_sizes
        except Exception:
            print('ERROR: could not parse --sizes, using defaults')
            matrix_sizes = default_sizes
    else:
        matrix_sizes = default_sizes
    
    # Get a single log directory for this run
    log_dir = get_available_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    print(f"Starting benchmark suite with {len(matrix_sizes)} different matrix sizes")
    print(f"Executable: {args.executable}")
    print(f"Logs will be saved to: {log_dir}")
    
    start_time = time.time()
    successful_runs = 0
    
    # Helper to normalize run method aliases into the full names expected by the binary
    def normalize_run_arg(run_arg):
        if run_arg is None:
            return None
        parts = [p.strip() for p in run_arg.split(',') if p.strip()]
        normalized = []
        for p in parts:
            key = p.lower()
            if key in ALIASES:
                normalized.append(ALIASES[key])
            else:
                # If user supplied full name, accept it (case sensitive pass-through)
                normalized.append(p)
        return ','.join(normalized)

    normalized_run = normalize_run_arg(args.run)

    # Run benchmarks, possibly in parallel (different sizes may run concurrently)
    if args.jobs and args.jobs > 1:
        max_workers = args.jobs
        print(f"Running up to {max_workers} concurrent benchmark jobs")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(run_benchmark, args.executable, size, log_dir, normalized_run): size for size in matrix_sizes}
            for fut in concurrent.futures.as_completed(futures):
                size = futures[fut]
                try:
                    ok = fut.result()
                    if ok:
                        successful_runs += 1
                except Exception as e:
                    print(f"Benchmark for size {size} failed: {e}")
    else:
        for size in matrix_sizes:
            if run_benchmark(args.executable, size, log_dir, normalized_run):
                successful_runs += 1
            else:
                assert(0)
            print("-" * 50)
    
    total_time = time.time() - start_time
    print(f"Benchmark suite completed: {successful_runs}/{len(matrix_sizes)} successful runs")
    print(f"Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
