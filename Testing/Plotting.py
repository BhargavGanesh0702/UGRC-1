import matplotlib.pyplot as plt

# Data
#1 - prml
#2 - real estate
#3 - generated
testcases = [1, 2, 3]
cuda_cpp = [21.3087, 65.5483, 28.6529]
pytorch_cpu = [15.6081, 47.2300, 21.4789]
pytorch_gpu = [29.3014, 91.1746, 35.8258]

plt.figure(figsize=(8, 5))
plt.plot(testcases, cuda_cpp, marker='o', label='CUDA (C++)')
plt.plot(testcases, pytorch_cpu, marker='o', label='PyTorch CPU')
plt.plot(testcases, pytorch_gpu, marker='o', label='PyTorch GPU')
plt.title('Best Case Time Comparison')
plt.xlabel('Testcase Number')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True)
plt.xticks(testcases)
plt.tight_layout()
plt.savefig('best_case_time_comparison.png')
plt.show()
