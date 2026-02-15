#!/usr/bin/env python3

import subprocess
import os
import time
import argparse
from datetime import datetime
import concurrent.futures
import re

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
    'local': 'BlockLocal-StackTranspose',
    'naive-par': 'Naive-ijkLoop-Parallel',
    'tiled-par': 'BlockTiled-CacheAware-Parallel',
    'tbb': 'Parallel-SIMD-TBB',
    'cuda': 'CUDA-Naive'
}

# Build reverse map: full method name -> alias (pick the first alias if duplicates)
FULL_TO_ALIAS = {}
for a, full in ALIASES.items():
    if full not in FULL_TO_ALIAS:
        FULL_TO_ALIAS[full] = a

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

def run_benchmark(executable_path, matrix_size, log_dir, run_methods=None, baseline=None):
    """Run the matrix multiplication benchmark with the specified size.

    If `run_methods` is provided it is forwarded as `--run=...` to the executable.
    If `baseline` is provided it is forwarded as `--baseline=...` to the executable.
    """
    print(f"Running benchmark with matrix size {matrix_size}x{matrix_size}...")
    
    # Use microsecond resolution to avoid collisions when runs are close in time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Helper to create a short, safe label from the requested methods using aliases when available
    def _sanitize_token(tok):
        # replace non-alphanumeric with underscore
        s = re.sub(r'[^A-Za-z0-9\-]', '_', tok)
        # collapse multiple underscores
        s = re.sub(r'_+', '_', s).strip('_')
        return s[:100]

    if run_methods:
        parts = [p.strip() for p in run_methods.split(',') if p.strip()]
        alias_parts = []
        for full in parts:
            if full in FULL_TO_ALIAS:
                alias_parts.append(FULL_TO_ALIAS[full])
            else:
                alias_parts.append(_sanitize_token(full))
        methods_label = '+'.join(alias_parts)
    else:
        methods_label = 'all'

    if baseline:
        # prefer alias for baseline when available
        baseline_label = FULL_TO_ALIAS.get(baseline, _sanitize_token(baseline))
        methods_label = f"{methods_label}_b={baseline_label}"

    log_file = f"{log_dir}/matmul_bench_{matrix_size}_{methods_label}_{timestamp}.log"
    
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
            if baseline:
                # forward baseline selection to the executable
                cmd.append(f"--baseline={baseline}")

            process = subprocess.run(cmd,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True,
                                     check=True)

            f.write(process.stdout)

        elapsed = time.time() - start_time
        print(f"[OK] Completed in {elapsed:.2f} seconds. Log saved to: {log_file}")
        return True, process.stdout
    
    except subprocess.CalledProcessError as e:
        with open(log_file, 'a') as f:
            f.write(f"\nERROR: Process failed with exit code {e.returncode}\n")
            if e.stdout:
                f.write(e.stdout)
        
        print(f"[ERROR] Failed with exit code {e.returncode}. Log saved to: {log_file}")
        return False, (e.stdout if e.stdout else "")
    
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"\nERROR: {str(e)}\n")
        
        print(f"[ERROR] Exception occurred: {str(e)}. Log saved to: {log_file}")
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description='Run matrix multiplication benchmarks with various sizes')
    default_exec = './bench-matmul'
    # Try to find the executable in common build locations
    possible_paths = [
        'build/Release/bench-matmul.exe',
        'build/Debug/bench-matmul.exe',
        'build/bench-matmul.exe',
        'build/bench-matmul',
        './bench-matmul.exe',
        './bench-matmul'
    ]
    for p in possible_paths:
        if os.path.exists(p):
            default_exec = p
            break

    parser.add_argument('--executable', default=default_exec,
                        help=f'Path to the benchmark executable (default: {default_exec})')
    parser.add_argument('--run', default=None,
                        help='Comma-separated list of multiplication methods to run (passed through to the executable as --run=name1,name2). Aliases supported.')
    parser.add_argument('--baseline', default=None,
                        help='(Optional) Baseline method name or alias to forward to the executable as --baseline. Example: --baseline=naive')
    parser.add_argument('--sizes', default=None,
                        help='Optional comma-separated list of matrix sizes to run (overrides defaults). Example: --sizes=64,128')
    parser.add_argument('--list', action='store_true', help='Query the executable for available multiplication methods and exit')
    parser.add_argument('-j', '--jobs', type=int, default=1, help='Number of concurrent runs (default: 1)')
    parser.add_argument('--compare', default=None,
                        help='Compare two or more comma-separated methods (aliases allowed). Example: --compare=naive,tiled-par')
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

    # Helper: query executable for available method names (calls --list)
    def get_available_methods(executable_path):
        try:
            res = subprocess.run([executable_path, '--list'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, check=True)
            out = res.stdout.splitlines()
            methods = set()
            for line in out:
                line = line.strip()
                if not line:
                    continue
                # lines that are method names are not prefixed with brackets; some lines may be logs
                if line.startswith('Available multiplication methods:'):
                    continue
                if line.startswith('['):
                    # skip log lines
                    continue
                # accept this line as a method if it looks like a name (contains letters and maybe hyphens)
                methods.add(line)
            return methods
        except Exception:
            return set()

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
    # Normalize baseline alias (single method)
    def normalize_single_method(token):
        if token is None:
            return None
        key = token.strip().lower()
        if not key:
            return None
        if key in ALIASES:
            return ALIASES[key]
        return token.strip()

    normalized_baseline = normalize_single_method(args.baseline)

    # If compare mode requested, run side-by-side timing comparisons and exit
    if args.compare:
        parts = [p.strip() for p in args.compare.split(',') if p.strip()]
        if len(parts) < 2:
            print('ERROR: --compare requires at least two method names separated by commas')
            return
        # Normalize each requested method separately
        methods = [normalize_run_arg(p) for p in parts]

        # Do not automatically remove the naive baseline; include whatever the user requested.
        if len(methods) < 2:
            print('ERROR: --compare requires at least two method names to compare')
            return

        # Validate methods against executable
        available = get_available_methods(args.executable)
        if available:
            unknown = [m for m in methods if m not in available]
            if unknown:
                print('ERROR: The following methods are not available from the executable:')
                for u in unknown:
                    print(f'  - {u}')
                print('\nAvailable methods:')
                for m in sorted(available):
                    print(f'  {m}')
                return
        else:
            print('WARNING: Could not query available methods from executable; proceeding without validation')

        print(f"Comparing methods: {', '.join(methods)}")
        # We'll run each method once per size and parse the average execution time from stdout
        def extract_avg_time(output):
            # Look for the 'Average execution time for' line
            import re
            matches = re.findall(r"Average execution time for .*?: ([0-9]+\.?[0-9]*) ms", output)
            if matches:
                return float(matches[-1])
            # Fallback: try to find any numeric timing in ms
            matches = re.findall(r"([0-9]+\.?[0-9]*) ms", output)
            if matches:
                return float(matches[-1])
            return None

        results = []
        for size in matrix_sizes:
            print(f"\nSize: {size}")
            times = []
            for m in methods:
                ok, out = run_benchmark(args.executable, size, log_dir, m, normalized_baseline)
                if not ok:
                    print(f"Run failed for {m} at size {size}")
                    times.append(None)
                    continue
                t = extract_avg_time(out)
                times.append(t)

            results.append((size, times))
        # Build comparison table (times per method) and write to a file
        table_lines = []
        table_lines.append('\nComparison results (times in ms):')
        header = f"{'Size':>6} | " + ' | '.join([f"{m:30}" for m in methods])
        table_lines.append(header)
        table_lines.append('-' * (len(header)))
        for size, times in results:
            row = f"{size:6} | "
            for t in times:
                if t is None:
                    row += f"{'N/A':30} | "
                else:
                    row += f"{t:30.3f} | "
            table_lines.append(row)

        # Print to stdout
        for line in table_lines:
            print(line)

        # Also write the comparison table to a timestamped log file under a logs directory
        try:
            # Use the same log directory used for individual runs so compare
            # output appears alongside per-run logs instead of creating a
            # separate `logs_<n>` directory.
            out_log_dir = log_dir
            os.makedirs(out_log_dir, exist_ok=True)

            # create a short filename based on methods using aliases when available
            def _sanitize(tok):
                s = re.sub(r'[^A-Za-z0-9\-]', '_', tok)
                s = re.sub(r'_+', '_', s).strip('_')
                return s[:120]

            method_tokens = []
            for full in methods:
                # If methods were provided as normalized run args they may be comma-joined
                # handle that by splitting and picking aliases where possible
                if ',' in full:
                    parts = [p.strip() for p in full.split(',') if p.strip()]
                else:
                    parts = [full]
                for p in parts:
                    method_tokens.append(FULL_TO_ALIAS.get(p, _sanitize(p)))

            methods_label = '+'.join(method_tokens) if method_tokens else 'compare'
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            comp_file = f"{out_log_dir}/comparison_{methods_label}_{ts}.txt"
            with open(comp_file, 'w') as cf:
                cf.write('\n'.join(table_lines))

                # Write CSV file with the same data
                csv_file = f"{out_log_dir}/comparison_{methods_label}_{ts}.csv"
                with open(csv_file, 'w') as csvf:
                    # Write header row: Size, method1, method2, ...
                    csvf.write('Size')
                    for m in methods:
                        csvf.write(f',"{m}"')
                    csvf.write('\n')
                    # Write each row: size, time1, time2, ...
                    for size, times in results:
                        csvf.write(str(size))
                        for t in times:
                            if t is None:
                                csvf.write(',')
                            else:
                                csvf.write(f',{t:.3f}')
                        csvf.write('\n')

            print(f"Comparison table written to: {comp_file}")
            print(f"CSV comparison table written to: {csv_file}")
        except Exception as e:
            print(f"WARNING: failed to write comparison file: {e}")
        return

    # Validate requested methods (if any) against the executable's available methods
    if normalized_run:
        available = get_available_methods(args.executable)
        if available:
            requested = [r.strip() for r in normalized_run.split(',') if r.strip()]
            unknown = [r for r in requested if r not in available]
            if unknown:
                print('ERROR: The following methods are not available from the executable:')
                for u in unknown:
                    print(f'  - {u}')
                print('\nAvailable methods:')
                for m in sorted(available):
                    print(f'  {m}')
                return
        else:
            # Could not query methods; warn but continue
            print('WARNING: Could not query available methods from executable; skipping validation')

    # Run benchmarks, possibly in parallel (different sizes may run concurrently)
    if args.jobs and args.jobs > 1:
        max_workers = args.jobs
        print(f"Running up to {max_workers} concurrent benchmark jobs")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(run_benchmark, args.executable, size, log_dir, normalized_run, normalized_baseline): size for size in matrix_sizes}
            for fut in concurrent.futures.as_completed(futures):
                size = futures[fut]
                try:
                    ok, _ = fut.result()
                    if ok:
                        successful_runs += 1
                except Exception as e:
                    print(f"Benchmark for size {size} failed: {e}")
    else:
        for size in matrix_sizes:
            ok, _ = run_benchmark(args.executable, size, log_dir, normalized_run, normalized_baseline)
            if ok:
                successful_runs += 1
            else:
                print(f"Benchmark failed for size {size}; continuing with remaining sizes")
            print("-" * 50)
    
    total_time = time.time() - start_time
    print(f"Benchmark suite completed: {successful_runs}/{len(matrix_sizes)} successful runs")
    print(f"Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()

