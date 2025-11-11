# UGRC(CUDA based Parallel Framework for Feed Forward Neural Networks)

## Main.cu
1. You need to change the file paths in Main.cu to run for different testcases.
2. You can change network configuration by changing corresponding input and output feature sizes and vector that contains hidden layer sizes.

### Network class functions
1. `void init_Network()` - Initializes the network with given configurations.
2. `vector<float> Train()` - Given training examples with expected outputs, no of epochs and learning rate, It trains model using stochastic gradient descent.
3. `float* predict()` - After training, This function gives the predicted value for given input X.


### compilation
`nvcc -arch=sm_75 Main.cu AllKernels.cu MatMul.cu Layer.cu Network.cu -o main_exec`

### execution
`./main_exec`

## Testing
I wrote `AutoTest.py` in `Testing` folder to partially automate testing
Before executing `AutoTest.py` do the following
1. Decide upon a testcase inside `Testcases` (Use only scaled testcases. If you want to use new testcase scale the data using `data_scaling.py` in the same folder)
2. Change paths of train and test csv files in `Main.cu` and `AutoTest.py`
3. Change feature sizes in `Main.cu`
4. Decide upon model configuration (hidden layer sizes, no of hidden layers, no of epoch, learning rate) and use same in both `Main.cu` and `PyTorch.py`
5. change iterations variable in `AutoTest.py` if you want
5. Run `python3 Testing/AutoTest.py` from main folder.

This will run `./Main.cu`, `PyTorch.py` in cpu , `PyTorch.py` in gpu iterations no of times and prints avg timings, avg losses, avg epoch errors.

