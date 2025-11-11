import subprocess
import statistics

cuda_exec = "./matmul"
seq_exec = "./a.out"
test_files = ["MatMulTestInput1.txt", "MatMulTestInput2.txt", "MatMulTestInput3.txt"]
runs = 100

def run_program(cmd, infile):
    result = subprocess.run(cmd, input=open(infile, "r").read(),
                            capture_output=True, text=True, shell=True)
    return result.stdout.strip()

def parse_cuda_output(output):
    lines = output.splitlines()
    times = []
    for line in lines:
        try:
            t = float(line.strip().split()[-1])
            times.append(t)
        except:
            pass
    if len(times) == 4:
        return times 
    else:
        return [0, 0, 0, 0]

def parse_seq_output(output):
    """Parse 1 timing line from a.out output."""
    lines = output.splitlines()
    for line in lines:
        try:
            return float(line.strip().split()[-1]) 
        except:
            pass
    return 0.0

def average(lst):
    return sum(lst) / len(lst) if lst else 0.0

print("===== Benchmarking Matrix Multiplication =====\n")
for test in test_files:
    print(f"Running tests for {test} ...")

    cuda_times = {"pad": [], "kernel": [], "unpad": [], "total": []}
    seq_times = []

    for i in range(runs):
        cuda_out = run_program(cuda_exec, test)
        T1, T2, T3, T4 = parse_cuda_output(cuda_out)
        cuda_times["pad"].append(T1)
        cuda_times["kernel"].append(T2)
        cuda_times["unpad"].append(T3)
        cuda_times["total"].append(T4)

        seq_out = run_program(seq_exec, test)
        T5 = parse_seq_output(seq_out)
        seq_times.append(T5)

    avg_pad = average(cuda_times["pad"])
    avg_kernel = average(cuda_times["kernel"])
    avg_unpad = average(cuda_times["unpad"])
    avg_total_cuda = average(cuda_times["total"])
    avg_total_seq = average(seq_times)

    print(f"\nResults for {test}:")
    print(f"  CUDA Avg Padding Time     (T1): {avg_pad:.6f} ms")
    print(f"  CUDA Avg Kernel Time      (T2): {avg_kernel:.6f} ms")
    print(f"  CUDA Avg Unpadding Time   (T3): {avg_unpad:.6f} ms")
    print(f"  CUDA Avg Total Time       (T4): {avg_total_cuda:.6f} ms")
    print(f"  Sequential Avg Total Time (T5): {avg_total_seq:.6f} ms")
    print("-" * 60)

print("\n===== Benchmarking Complete =====")
