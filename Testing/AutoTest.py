import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

iterations = 10 
train_path = "Testcases/scaled_real_estate_train.csv"
test_path = "Testcases/scaled_real_estate_test.csv"
main_cu_executable = "./main_exec" 

configs = {
    "CUDA (C++)": ["./main_exec"],
    "PyTorch CPU": ["python3", "Testing/PyTorch.py", train_path, test_path, "cpu", "1"],
    "PyTorch GPU": ["python3", "Testing/PyTorch.py", train_path, test_path, "gpu", "1"]
}

def run_config(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    try:
        #format:
        #time
        #[error1, error2, ...]
        #mse
        training_time = float(lines[0])
        epoch_loss = [float(x) for x in lines[1].strip("[]").split(",")]
        mse_loss = float(lines[2])
        return {
            "training_time": training_time,
            "epoch_loss": epoch_loss,
            "test_mse_loss": mse_loss
        }
    except Exception as e:
        print(f"Wrong format")
        return None


summary = {}

for name, cmd in configs.items():
    print(f"\nRunning configuration: {name}")
    times, test_losses, errors = [], [], []
    for i in range(iterations):
        print(f"  Iteration {i+1}/{iterations}")
        res = run_config(cmd)
        if res:
            times.append(res["training_time"])
            test_losses.append(res["test_mse_loss"])
            errors.append(res["epoch_loss"])

    avg_time = np.mean(times)
    avg_mse = np.mean(test_losses)
    avg_error = np.mean(np.array(errors), axis=0).tolist()

    summary[name] = {
        "avg_time": avg_time,
        "avg_mse": avg_mse,
        "avg_error": avg_error
    }

print("\n=== Average Training Times (s) ===")
for name, stats in summary.items():
    print(f"{name:15}: {stats['avg_time']:.4f}")

print("\n=== Average Test MSE Loss ===")
for name, stats in summary.items():
    print(f"{name:15}: {stats['avg_mse']:.6f}")

plt.figure(figsize=(10,6))
for name, stats in summary.items():
    print(stats["avg_error"])
    plt.plot(stats["avg_error"], label=name)

plt.xlabel("Epoch")
plt.ylabel("Average Epoch Loss")
plt.title("Training Error vs Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("error_vs_epoch.png")
plt.show()
