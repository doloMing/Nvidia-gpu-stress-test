# GPU/CPU Stress Test Documentation

## Overview about this code

- nvidia_gpu_stress_test.py contains four GPU stress test modes: Matrix Mode, Simple Mode, Ray Tracing Mode, and Frequency Max Mode. Each mode is designed to stress different aspects of GPU performance and capabilities. In general, these modes provide comprehensive testing capabilities for various aspects of GPU performance. By selecting the appropriate mode based on the specific capabilities you wish to test, you can gain valuable insights into your GPU's performance characteristics. Whether you're interested in computational throughput, rendering performance, or thermal management, these modes provide targeted testing to meet your needs.
- intel_cpu_stress_test.py is a Python-based utility designed for stress testing Intel CPUs. It provides precise control over CPU utilization, supports multi-core testing, and offers detailed performance monitoring capabilities.
- amd_cpu_stress_test.py is adapted from the Intel CPU stress testing framework, with specific optimizations for AMD CPU architectures. You can use it on AMD CPUs just like how you use intel_cpu_stress_test.py on Intel CPUs.

# Multi-mode Nvidia GPU Stress Test

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
def matrix_stress(duration: int, target_percent: float, gpu_index: int = 0, log_file: str = None) -> Dict[int, dict]:
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
def simple_stress(stop_flag: list, target_percent: float, gpu_index: int, duration: int) -> dict:
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

## Mode Suitability for GPU Capability Testing

Each stress test mode is designed to evaluate specific aspects of GPU performance. Understanding which mode to use for different testing scenarios can help in effectively assessing GPU capabilities.

### 1. Matrix Mode

#### Suitable for Testing:
- **Computational Throughput**: Evaluates the raw processing power of the GPU by performing intensive matrix operations.
- **Memory Bandwidth**: Tests the ability of the GPU to handle large data transfers between memory and compute units.
- **Cache Efficiency**: Assesses how well the GPU utilizes its cache hierarchy during compute-heavy tasks.

#### Ideal Use Cases:
- Benchmarking GPU performance in scientific computing applications.
- Evaluating the efficiency of memory access patterns.
- Testing the impact of different memory configurations on performance.

### 2. Simple Mode

#### Suitable for Testing:
- **Basic Compute Stability**: Provides a straightforward way to test the stability of the GPU under load.
- **Thermal Management**: Monitors how the GPU handles heat generation during sustained compute tasks.
- **Power Consumption**: Evaluates the power efficiency of the GPU during basic operations.

#### Ideal Use Cases:
- Initial stability testing for new hardware setups.
- Thermal profiling to ensure adequate cooling solutions.
- Power draw analysis for energy efficiency studies.

### 3. Ray Tracing Mode

#### Suitable for Testing:
- **Rendering Performance**: Simulates real-world rendering workloads to test the GPU's ability to handle complex graphics tasks.
- **Memory Latency**: Evaluates the impact of random memory access patterns typical in ray tracing applications.
- **Compute and Memory Balance**: Tests the GPU's ability to balance compute and memory operations efficiently.

#### Ideal Use Cases:
- Benchmarking performance in graphics-intensive applications like gaming and 3D rendering.
- Testing the impact of different memory configurations on rendering performance.
- Evaluating the effectiveness of GPU architectures in handling ray tracing workloads.

### 4. Frequency Max Mode

#### Suitable for Testing:
- **Clock Stability**: Tests the GPU's ability to maintain high clock speeds under load without throttling.
- **Thermal Throttling**: Evaluates the GPU's thermal management capabilities and its impact on performance.
- **Power Delivery**: Assesses the power supply's ability to support the GPU at maximum frequency.

#### Ideal Use Cases:
- Stress testing for overclocking scenarios to ensure stability at higher frequencies.
- Thermal solution evaluation to prevent throttling during high-performance tasks.
- Power supply testing to ensure adequate delivery under peak load conditions.

# Intel CPU Stress Test

## Features
- Controllable CPU utilization (1-100%)
- Multi-core support with individual core selection
- Hyperthreading control
- Real-time performance monitoring
- Detailed test logging
- Progress visualization
- Configurable test duration
- Support for both physical and logical cores

## Requirements
- Python 3.6 or higher
- Required Python packages:
  - psutil
  - multiprocessing
  - argparse

## Usage
### Basic Command
```bash
python intel_cpu_stress_test.py [-h] [-d DURATION] [-t TARGET] [-c CORES [CORES ...]] 
                               [--disable-ht] [-o [OUTPUT]]
```

### Command Line Arguments
- `-h, --help`: Show help message
- `-d, --duration`: Test duration in seconds (default: 60)
- `-t, --target`: Target CPU usage percentage (default: 95)
- `-c, --cores`: Specific CPU cores to use (e.g., -c 0 2 4 6)
- `--disable-ht`: Disable hyperthreading (use only physical cores)
- `-o, --output`: Path to save the log file

### Example Commands
```bash
# Basic test with default settings
python intel_cpu_stress_test.py

# Test with custom duration and target load
python intel_cpu_stress_test.py -d 120 -t 80

# Test specific CPU cores
python intel_cpu_stress_test.py -c 0 2 4

# Test with hyperthreading disabled
python intel_cpu_stress_test.py --disable-ht

# Test with output logging
python intel_cpu_stress_test.py -o test_results.txt
```

## Test Modes and Scenarios

### 1. Full Load Test
Tests CPU at maximum capacity:
```bash
python intel_cpu_stress_test.py -t 100 -d 300
```
Use Case: Stability testing, thermal performance evaluation

### 2. Selective Core Test
Tests specific CPU cores:
```bash
python intel_cpu_stress_test.py -c 0 2 4 6 -t 90
```
Use Case: Core-specific performance analysis, thermal distribution testing

### 3. Physical Cores Only
Tests without hyperthreading:
```bash
python intel_cpu_stress_test.py --disable-ht -t 80
```
Use Case: Base CPU performance testing, SMT impact analysis

### 4. Endurance Test
Long-duration testing:
```bash
python intel_cpu_stress_test.py -d 3600 -t 70
```
Use Case: Stability verification, thermal throttling analysis

## Output and Monitoring

### Real-time Display
During the test, the following information is displayed in real-time:
- Progress bar showing current CPU utilization
- Current overall CPU usage
- Target CPU usage
- Per-core utilization
- Remaining test time

Example output:
```
[==========-----] | Current: 50.5% | Target: 50.0% | Core 0: 49.8% | Core 1: 51.2% | Time: 45s
```

### Test Results
After completion, the tool displays:
- Average CPU usage during the test
- Per-core average utilization statistics
- Test duration and configuration details

Example results:
```
Average CPU Usage during test: 94.8%

Per-core average statistics during test:
Core 0: 95.2%
Core 1: 94.7%
Core 2: 94.9%
Core 3: 94.5%

CPU stress test has been completed.
```

## Advanced Usage Scenarios

### 1. Multi-Phase Testing
```bash
# Phase 1: All cores, high load
python intel_cpu_stress_test.py -d 300 -t 90 -o phase1.txt

# Phase 2: Physical cores only
python intel_cpu_stress_test.py -d 300 -t 60 --disable-ht -o phase2.txt

# Phase 3: Specific cores
python intel_cpu_stress_test.py -d 300 -t 75 -c 0 2 4 -o phase3.txt
```

### 2. Performance Profiling
```bash
# Single core maximum performance
python intel_cpu_stress_test.py -c 0 -t 100 -o single_core.txt

# Dual-core balanced test
python intel_cpu_stress_test.py -c 0 1 -t 50 -d 300 -o dual_core.txt
```

# AMD CPU Stress Test

## AMD-Specific Features and Adaptations

### 1. Architectural Differences
- **SMT vs Hyperthreading**: AMD uses Simultaneous Multi-Threading (SMT) instead of Intel's Hyperthreading
- **CCX (Core Complex)**: AMD CPUs use CCX architecture, which affects core grouping and cache sharing
- **Precision Boost**: AMD's equivalent to Intel's Turbo Boost, with different behavior patterns
- **Core Layout**: Different physical to logical core mapping compared to Intel

### 2. Implementation Adaptations

#### Core Topology Handling
```python
def get_cpu_topology():
    """
    Get CPU topology information including physical and logical cores
    For AMD CPUs, this includes CCX (Core Complex) information if available
    """
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    return physical_cores, logical_cores
```

#### SMT Control
```python
if disable_smt:
    # Use only physical cores when SMT is disabled
    available_cores = list(range(physical_cores))
else:
    # Use all logical cores when SMT is enabled
    available_cores = list(range(logical_cores))
```

## AMD-Specific Test Scenarios

### 1. CCX-Aware Testing
```bash
# Test cores within same CCX
python amd_cpu_stress_test.py -c 0 1 2 3 -t 100

# Test cross-CCX performance
python amd_cpu_stress_test.py -c 0 4 -t 100
```

### 2. Precision Boost Analysis
```bash
# Single-core boost behavior
python amd_cpu_stress_test.py -c 0 -t 100 -d 300

# Multi-core boost scaling
for cores in {1..8}; do
    python amd_cpu_stress_test.py -c $(seq -s ' ' 0 $((cores-1))) -t 100 -d 300
done
```

### 3. SMT Performance Testing
```bash
# Test with SMT enabled
python amd_cpu_stress_test.py -t 100 -d 300

# Test with SMT disabled
python amd_cpu_stress_test.py --disable-smt -t 100 -d 300
```

## AMD-Specific Considerations

### 1. CCX Architecture
- Cores within same CCX share L3 cache
- Cross-CCX communication has different latency characteristics
- Consider CCX boundaries when selecting cores for testing

### 2. Precision Boost Behavior
- Different boost algorithm compared to Intel
- Temperature and power limits affect boost behavior differently
- Consider monitoring tools like `ryzen_monitor` for detailed boost analysis

### 3. Power Management
- Different power states compared to Intel
- Package power tracking differs from Intel's implementation
- Consider using `ryzen_smu` for detailed power monitoring

## Best Practices for AMD Testing

### 1. CCX-Aware Core Selection
```bash
# For Ryzen CPUs with 2 CCXs (example)
# Test first CCX
python amd_cpu_stress_test.py -c 0 1 2 3 -o ccx1_test.txt

# Test second CCX
python amd_cpu_stress_test.py -c 4 5 6 7 -o ccx2_test.txt

# Cross-CCX test
python amd_cpu_stress_test.py -c 0 4 -o cross_ccx_test.txt
```

### 2. Thermal Considerations
- AMD CPUs often have different thermal characteristics
- Consider chiplet design in newer AMD processors
- Monitor per-CCX temperatures when available

### 3. Performance Monitoring
```bash
# Gradual load increase for boost behavior analysis
for load in 30 50 70 90 100; do
    python amd_cpu_stress_test.py -d 300 -t $load -o boost_${load}.txt
done
```

## Advanced Test Examples

### 1. CCX Performance Analysis
```bash
#!/bin/bash
# Test inter-CCX and intra-CCX performance

# Intra-CCX test (cores in same CCX)
python amd_cpu_stress_test.py -c 0 1 -t 100 -d 300 -o intra_ccx.txt

# Inter-CCX test (cores in different CCXs)
python amd_cpu_stress_test.py -c 0 4 -t 100 -d 300 -o inter_ccx.txt

# Compare results for CCX impact analysis
```

### 2. Precision Boost Optimization
```bash
#!/bin/bash
# Analyze Precision Boost behavior under different conditions

# Single-core boost
python amd_cpu_stress_test.py -c 0 -t 100 -d 300 -o single_boost.txt

# All-core boost
python amd_cpu_stress_test.py -t 100 -d 300 -o all_core_boost.txt

# Varying core count for boost scaling analysis
```

### 3. Memory Controller Stress Test
```bash
# High-load test focusing on memory controller
python amd_cpu_stress_test.py -d 900 -t 100 -o memory_controller.txt
```

## Compatibility Notes

### 1. Processor Families
- Ryzen 1000-7000 series supported
- Threadripper requires additional core mapping consideration
- EPYC servers may need different core selection strategies

### 2. Operating Systems
- Linux: Full support with AMD monitoring tools
- Windows: Basic functionality with limited low-level access
- Consider OS-specific thread scheduling differences
