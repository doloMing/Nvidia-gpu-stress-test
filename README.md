# GPU Stress Test Modes Documentation

## Overview about this code

This documentation details four GPU stress test modes implemented in the code: Matrix Mode, Simple Mode, Ray Tracing Mode, and Frequency Max Mode. Each mode is designed to stress different aspects of GPU performance and capabilities. In general, these modes provide comprehensive testing capabilities for various aspects of GPU performance. By understanding the load distribution and implementation details, users can effectively utilize these tests to evaluate and optimize their GPU hardware.

## Main Function Parameters

The `main` function in the GPU stress test tool is designed to handle command-line arguments that configure the test's behavior. Understanding these parameters is crucial for effectively using the tool.

### Command-Line Arguments

#### 1. `-d`, `--duration`
- **Description**: Specifies the duration of the stress test in seconds.
- **Default Value**: 60 seconds
- **Usage**: Determines how long the test will run. Longer durations provide more comprehensive data but require more time.
- **Example**: `-d 120` runs the test for 120 seconds.

#### 2. `-t`, `--target`
- **Description**: Sets the target GPU usage percentage.
- **Default Value**: 50%
- **Usage**: Adjusts the load applied to the GPU. Useful for testing different levels of stress.
- **Example**: `-t 80` targets 80% GPU utilization.

#### 3. `-p`, `--parallel`
- **Description**: Enables parallel mode, allowing tests to run on multiple GPUs simultaneously.
- **Usage**: Useful for systems with multiple GPUs, enabling concurrent testing.
- **Example**: `-p` runs tests in parallel mode.

#### 4. `-g`, `--gpus`
- **Description**: Specifies a comma-separated list of GPU indices to test.
- **Usage**: Allows selection of specific GPUs for testing, useful in multi-GPU systems.
- **Example**: `-g 0,1` tests GPUs with indices 0 and 1.

#### 5. `-l`, `--loads`
- **Description**: Provides a comma-separated list of target loads for each GPU.
- **Usage**: Sets different target loads for each specified GPU, allowing customized stress levels.
- **Example**: `-l 60,70` sets 60% load for the first GPU and 70% for the second.

#### 6. `-o`, `--output`
- **Description**: Specifies the path to save the log file.
- **Default Value**: None (logs are not saved by default)
- **Usage**: Enables logging of test results to a file for later analysis.
- **Example**: `-o ./logs` saves logs to the `logs` directory.

#### 7. `-m`, `--mode`
- **Description**: Selects the test mode to run.
- **Choices**: `matrix`, `simple`, `ray`, `frequency-max`
- **Default Value**: `frequency-max`
- **Usage**: Determines the type of stress test to perform, each focusing on different GPU aspects.
- **Example**: `-m ray` runs the ray tracing stress test.

### Example Usage

```bash
# Run a matrix mode test for 90 seconds targeting 75% GPU usage on GPU 0
python gpu_stress_test.py -m matrix -d 90 -t 75 -g 0

# Run a parallel simple mode test on GPUs 0 and 1 with different loads
python gpu_stress_test.py -m simple -p -g 0,1 -l 50,70

# Run a frequency max test for 120 seconds and save the output to a log file
python gpu_stress_test.py -m frequency-max -d 120 -o ./logs
```

## Test Modes in Detail

### 1. Matrix Mode (matrix_gpu_stress_test)

#### Purpose
Tests GPU's computational capabilities through matrix operations, primarily focusing on CUDA cores and memory bandwidth.

#### Load Distribution
- **Primary Load**: CUDA cores (80-90%)
- **Secondary Load**: Memory bandwidth (40-60%)
- **Cache Utilization**: High (L1 and L2)
- **Memory Access Pattern**: Strided, optimized for coalescing

#### Implementation Details
```python
def matrix_gpu_stress_test(duration: int, target_percent: float, gpu_index: int, log_file: str = None) -> dict:
    """
    Parameters:
    - duration: Test duration in seconds
    - target_percent: Target GPU utilization (1-100)
    - gpu_index: GPU device index
    - log_file: Optional log file path
    
    Returns:
    - Dictionary containing test metrics
    """
```

#### Key Features
1. **Dynamic Matrix Size Adjustment**
   - Automatically scales based on GPU memory
   - Adjusts for optimal cache utilization
   - Prevents OOM errors

2. **Load Control Mechanism**
   - PID controller for utilization targeting
   - Adaptive sleep intervals
   - Real-time load adjustment

3. **Performance Metrics**
   - GPU utilization
   - Memory bandwidth
   - Cache hit rates
   - Operation throughput

#### Usage Example
```bash
# Basic matrix test
python gpu_stress_test.py -m matrix -d 60 -t 80

# Multi-GPU matrix test
python gpu_stress_test.py -m matrix -d 60 -g 0,1 -t 80 -p
```

### 2. Simple Mode (simple_gpu_stress)

#### Purpose
Provides basic GPU stress testing through simple compute operations, ideal for initial testing and stability verification.

#### Load Distribution
- **Primary Load**: CUDA cores (70-80%)
- **Secondary Load**: Memory bandwidth (20-30%)
- **Cache Utilization**: Medium
- **Power Draw**: Moderate and stable

#### Implementation Details
```python
def simple_gpu_stress(stop_flag: list, target_percent: float, gpu_index: int, duration: int) -> dict:
    """
    Parameters:
    - stop_flag: List containing stop condition
    - target_percent: Target GPU utilization (1-100)
    - gpu_index: GPU device index
    - duration: Test duration in seconds
    
    Returns:
    - Dictionary containing test metrics
    """
```

#### Key Features
1. **Simplified Operation Set**
   - Basic arithmetic operations
   - Predictable load patterns
   - Low overhead monitoring

2. **Resource Management**
   - Efficient memory allocation
   - Automatic cleanup
   - Error recovery

3. **Stability Features**
   - Thermal monitoring
   - Power draw tracking
   - Utilization smoothing

#### Usage Example
```bash
# Basic simple test
python gpu_stress_test.py -m simple -d 60 -t 70

# Extended simple test with logging
python gpu_stress_test.py -m simple -d 300 -t 80 -o ./logs
```

### 3. Ray Tracing Mode (ray_tracing_stress)

#### Purpose
Simulates ray tracing workloads to test GPU's rendering capabilities, focusing on compute and memory access patterns.

#### Load Distribution
- **Primary Load**: Compute units (80-90%)
- **Secondary Load**: Memory bandwidth (30-50%)
- **Cache Utilization**: High
- **Memory Access Pattern**: Random, high latency

#### Implementation Details
```python
def ray_tracing_stress(stop_flag: list, target_percent: float, gpu_index: int, duration: int) -> dict:
    """
    Parameters:
    - stop_flag: List containing stop condition
    - target_percent: Target GPU utilization (1-100)
    - gpu_index: GPU device index
    - duration: Test duration in seconds
    
    Returns:
    - Dictionary containing test metrics
    """
```

#### Key Features
1. **Ray Tracing Simulation**
   - Simulates light paths and reflections
   - High computational intensity
   - Complex memory access patterns

2. **Dynamic Load Balancing**
   - Adjusts workload based on GPU performance
   - Real-time feedback loop
   - Utilization targeting

3. **Performance Metrics**
   - Frame rendering time
   - Ray tracing throughput
   - Memory latency

#### Usage Example
```bash
# Basic ray tracing test
python gpu_stress_test.py -m ray -d 60 -t 85

# Parallel ray tracing test on multiple GPUs
python gpu_stress_test.py -m ray -d 60 -g 0,1 -t 85 -p
```

### 4. Frequency Max Mode (frequency_stress)

#### Purpose
Maximizes GPU frequency to test stability and performance under high clock rates.

#### Load Distribution
- **Primary Load**: Clock management (100%)
- **Secondary Load**: Power consumption (high)
- **Thermal Impact**: Significant, requires monitoring

#### Implementation Details
```python
def frequency_stress(stop_flag: list, target_percent: float, gpu_index: int, duration: int) -> dict:
    """
    Parameters:
    - stop_flag: List containing stop condition
    - target_percent: Target GPU frequency percentage (ignored in max mode)
    - gpu_index: GPU device index
    - duration: Test duration in seconds
    
    Returns:
    - Dictionary containing test metrics
    """
```

#### Key Features
1. **Frequency Maximization**
   - Pushes GPU to maximum clock rates
   - Monitors stability and performance
   - Identifies thermal throttling

2. **Thermal Management**
   - Real-time temperature monitoring
   - Automatic throttling detection
   - Cooling system evaluation

3. **Performance Metrics**
   - Maximum frequency achieved
   - Stability under load
   - Power draw and efficiency

#### Usage Example
```bash
# Basic frequency max test
python gpu_stress_test.py -m frequency-max -d 60

# Extended frequency max test with logging
python gpu_stress_test.py -m frequency-max -d 120 -o ./logs
```

