import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import time

if len(sys.argv) != 5:
    print("Usage: python3 PyTorch.py <train.csv> <test.csv> <cpu/gpu> <batch_size>")
    sys.exit(1)

train_path = sys.argv[1]
test_path = sys.argv[2]
device_arg = sys.argv[3].lower()
batch_size = int(sys.argv[4])

if device_arg == "gpu" and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
X_cols = [col for col in train_df.columns if col.startswith('X')]
Y_cols = [col for col in train_df.columns if col.startswith('Y')]
X_train = train_df[X_cols].values
Y_train = train_df[Y_cols].values 
X_test = test_df[X_cols].values
Y_test = test_df[Y_cols].values  

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

class Net(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.tanh = nn.Tanh()
        for layer in [self.fc1, self.fc3]:
            nn.init.uniform_(layer.weight, -0.5, 0.5)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

input_dim = X_train.shape[1] 
hidden1 = 20
hidden2 = 10
output_dim = Y_train.shape[1] 

model = Net(input_dim, hidden1, hidden2, output_dim).to(device)
criterion = nn.MSELoss()

learning_rate = 0.01
epochs = 250
N = X_train.shape[0]
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
start_time = time.time()
error_list = []
for epoch in range(epochs):
    indices = torch.randperm(N, device=device)
    epoch_loss = 0.0

    for i in range(0, N, batch_size):
        batch_idx = indices[i:i+batch_size]
        x = X_train[batch_idx]
        y = Y_train[batch_idx]

        optimizer.zero_grad() 
        output = model(x)
        loss = criterion(output, y)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

    epoch_loss /= (N // batch_size)
    error_list.append(epoch_loss)
    # if ((epoch + 1) % 50 == 0) or (epoch + 1 == epochs):
        # print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {epoch_loss:.6f}")

end_time = time.time()
training_time = end_time - start_time
print(f"{training_time:.3f}")

formatted_errors = "[" + ", ".join(f"{e:.6f}" for e in error_list) + "]"
print(formatted_errors)
model.eval()
with torch.no_grad():
    preds = model(X_test)
    test_loss = criterion(preds, Y_test)
    print(f"{test_loss.item():.6f}")