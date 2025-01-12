# GPU Stress Test Modes Documentation

## Overview about this code

This documentation details four GPU stress test modes implemented in the code: Matrix Mode, Simple Mode, Ray Tracing Mode, and Frequency Max Mode. Each mode is designed to stress different aspects of GPU performance and capabilities. In general, these modes provide comprehensive testing capabilities for various aspects of GPU performance. By understanding the load distribution and implementation details, users can effectively utilize these tests to evaluate and optimize their GPU hardware.

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

