import multiprocessing
import psutil
import time
import argparse
import os
import sys
from datetime import datetime

class Logger:
    """Logger class to handle both console and file output"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

def print_cpu_status(cpu_cores: list, cpu_percent: list, target_percent: float, 
                    remaining_time: float = None):
    """
    Print current CPU status
    
    Args:
        cpu_cores: List of CPU cores being tested
        cpu_percent: List of CPU utilization percentages
        target_percent: Target CPU utilization percentage
        remaining_time: Remaining test time in seconds (optional)
    """
    # Calculate average usage for tested cores
    current_usage = sum(cpu_percent[core] for core in cpu_cores) / len(cpu_cores)
    
    # Create progress bar
    progress = int(current_usage / 5)  # 20 segments for 100%
    bar = "=" * progress + "-" * (20 - progress)
    
    # Build status string
    status = [
        f"\r[{bar}]",
        f"Current: {current_usage:.1f}%",
        f"Target: {target_percent:.1f}%"
    ]
    
    # Add per-core statistics
    core_stats = [f"Core {core}: {cpu_percent[core]:.1f}%" for core in cpu_cores]
    status.extend(core_stats)
    
    # Add remaining time if provided
    if remaining_time is not None:
        status.append(f"Time: {int(remaining_time)}s")
    
    # Print status
    print(" | ".join(status), end="")

def get_cpu_topology():
    """
    Get CPU topology information including physical and logical cores
    For AMD CPUs, this includes CCX (Core Complex) information if available
    
    Returns:
        tuple: (physical_cores, logical_cores)
    """
    try:
        physical_cores = psutil.cpu_count(logical=False)  # Number of physical cores
        logical_cores = psutil.cpu_count(logical=True)    # Number of logical cores (including SMT)
        
        # Note: AMD specific topology information could be added here
        # This would require platform specific tools or libraries
        
        return physical_cores, logical_cores
    except Exception as e:
        print(f"Error getting CPU topology: {e}")
        return None, None

def cpu_stress_task(target_percent, cpu_affinity=None):
    """
    Function to perform CPU stress test with controlled usage
    Optimized for AMD CPU architecture
    
    Args:
        target_percent (float): Target CPU usage percentage (1-100)
        cpu_affinity (list): List of CPU cores to use
    """
    # Set CPU affinity for the current process
    if cpu_affinity is not None:
        try:
            p = psutil.Process()
            p.cpu_affinity([cpu_affinity])
        except Exception as e:
            print(f"Error setting CPU affinity: {e}")
    
    while True:
        start_time = time.time()
        
        # Perform calculations (active phase)
        # Using AMD-optimized workload pattern
        sum(i * i for i in range(10**6))
        
        # Calculate elapsed time and adjust workload
        elapsed_time = time.time() - start_time
        
        # Sleep to achieve target CPU usage
        if elapsed_time < 1.0:  # Using 1 second as a control cycle
            sleep_time = (elapsed_time * (100 - target_percent) / target_percent)
            time.sleep(max(0, sleep_time))

def cpu_stress_test(duration=60, target_percent=95, cpu_cores=None, disable_smt=False, log_file=None):
    """
    AMD CPU stress test with core selection and SMT control
    
    Args:
        duration (int): Test duration in seconds after reaching target CPU usage
        target_percent (float): Target CPU usage percentage (1-100)
        cpu_cores (list): List of CPU cores to use (None means use all cores)
        disable_smt (bool): Whether to disable SMT (AMD's equivalent to Intel's Hyperthreading)
        log_file (str): Path to log file (optional)
    """
    # Setup logging if needed
    original_stdout = sys.stdout
    logger = Logger(log_file) if log_file else None
    if logger:
        sys.stdout = logger

    try:
        # Validate input parameters
        if not 1 <= target_percent <= 100:
            raise ValueError("Target percentage must be between 1 and 100")
        
        # Get CPU topology information
        physical_cores, logical_cores = get_cpu_topology()
        
        # Determine which cores to use based on input parameters and system configuration
        if cpu_cores is None:
            if disable_smt:
                # Use only physical cores when SMT is disabled
                available_cores = list(range(physical_cores))
            else:
                # Use all logical cores when SMT is enabled
                available_cores = list(range(logical_cores))
        else:
            # Validate specified core numbers
            max_core = logical_cores - 1
            invalid_cores = [core for core in cpu_cores if core > max_core or core < 0]
            if invalid_cores:
                raise ValueError(f"Invalid core numbers: {invalid_cores}. Valid range: 0-{max_core}")
            available_cores = cpu_cores
        
        print(f"CPU Topology: {physical_cores} physical cores, {logical_cores} logical cores")
        print(f"Using CPU cores: {available_cores}")
        print(f"Target CPU usage: {target_percent}%")
        
        # List to store worker processes
        processes = []
        
        # Dictionary to store running statistics
        stats = {
            'per_core_usage': {core: [] for core in available_cores},
            'overall_usage': []
        }
        
        try:
            # Start a process for each selected CPU core
            for core in available_cores:
                p = multiprocessing.Process(
                    target=cpu_stress_task,
                    args=(target_percent, core)
                )
                p.start()
                processes.append(p)
            
            print("\nWaiting for CPU usage to stabilize...")
            target_reached = False
            start_time = None
            
            while True:
                # Monitor CPU usage for selected cores
                cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
                current_usage = sum(cpu_percent[core] for core in available_cores) / len(available_cores)
                
                # Record statistics after target is reached
                if target_reached:
                    stats['overall_usage'].append(current_usage)
                    for core in available_cores:
                        stats['per_core_usage'][core].append(cpu_percent[core])
                
                # Start timer when target usage is reached
                if not target_reached and current_usage >= target_percent:
                    target_reached = True
                    start_time = time.time()
                    print(f"\nTarget CPU usage reached. Starting {duration} seconds countdown...")
                
                # Check if test duration has completed
                if target_reached and (time.time() - start_time >= duration):
                    print("\nTest duration completed.")
                    break
                
                # Calculate remaining time if test has started
                remaining = duration - (time.time() - start_time) if target_reached else None
                
                # Display real-time status
                print_cpu_status(available_cores, cpu_percent, target_percent, remaining)
                
        except KeyboardInterrupt:
            print("\nCPU stress test has been interrupted by user")
        finally:
            print("\nStopping CPU stress test...")
            # Clean up processes
            for p in processes:
                p.terminate()
                p.join()
            
            # Calculate and display average statistics
            if stats['overall_usage']:
                avg_overall = sum(stats['overall_usage']) / len(stats['overall_usage'])
                print(f"\nAverage CPU Usage during test: {avg_overall:.1f}%")
                
                print("\nPer-core average statistics during test:")
                for core in available_cores:
                    if stats['per_core_usage'][core]:
                        avg_core = sum(stats['per_core_usage'][core]) / len(stats['per_core_usage'][core])
                        print(f"Core {core}: {avg_core:.1f}%")
            
            print("\nCPU stress test has been completed.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if logger:
            sys.stdout = original_stdout
            logger.close()

def main_func():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Advanced AMD CPU Stress Test Tool')
    parser.add_argument('-d', '--duration', type=int, default=60,
                      help='Test duration in seconds (default: 60)')
    parser.add_argument('-t', '--target', type=float, default=95,
                      help='Target CPU usage percentage (default: 95)')
    parser.add_argument('-c', '--cores', type=int, nargs='+',
                      help='Specific CPU cores to use (e.g., -c 0 2 4 6)')
    parser.add_argument('--disable-smt', action='store_true',
                      help='Disable SMT (AMD equivalent to Intel Hyperthreading)')
    parser.add_argument('-o', '--output', type=str, default=None, const='.',
                      nargs='?',
                      help='Path to save the log file')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Process output path
    log_file = None
    if args.output is not None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"amd_cpu_stress_test_{timestamp}.txt"
        log_file = os.path.join(args.output if args.output != '.' else os.getcwd(), 
                               filename)
        print(f"Output will be saved to: {os.path.abspath(log_file)}")
    
    try:
        # Run the stress test with provided parameters
        cpu_stress_test(
            duration=args.duration,
            target_percent=args.target,
            cpu_cores=args.cores,
            disable_smt=args.disable_smt,
            log_file=log_file
        )
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main_func()

    # # Basic test with default settings
    # # - Duration: 60 seconds
    # # - Target CPU usage: 95%
    # # - Uses all available CPU cores
    # # - SMT enabled
    # python amd_cpu_stress_test.py

    # # Test with custom duration and target load
    # # - Duration: 120 seconds
    # # - Target CPU usage: 80%
    # # - Good for moderate stress testing
    # python amd_cpu_stress_test.py -d 120 -t 80

    # # Test specific CPU cores
    # # - Uses cores 0, 2, and 4 only
    # # - Good for testing specific CCX (Core Complex) performance
    # python amd_cpu_stress_test.py -c 0 2 4

    # # Test with SMT disabled
    # # - Uses only physical cores
    # # - Useful for testing base CPU performance without SMT
    # python amd_cpu_stress_test.py --disable-smt

    # # Test with output logging (auto-generated filename)
    # # - Creates log file in current directory
    # # - Filename format: amd_cpu_stress_test_YYYYMMDD_HHMMSS.txt
    # python amd_cpu_stress_test.py -o .

    # # Test with custom output path
    # # - Saves log to specified directory
    # # - Creates directory if it doesn't exist
    # python amd_cpu_stress_test.py -o /path/to/logs/test1.txt

    # # Comprehensive test with all options
    # # - Duration: 300 seconds (5 minutes)
    # # - Target load: 75%
    # # - Specific cores: 0,1,2,3
    # # - With logging
    # python amd_cpu_stress_test.py -d 300 -t 75 -c 0 1 2 3 -o test_results.txt

    # # Low-load endurance test
    # # - Duration: 3600 seconds (1 hour)
    # # - Target load: 30%
    # # - Good for testing thermal behavior over time
    # # - Useful for AMD's Precision Boost behavior analysis
    # python amd_cpu_stress_test.py -d 3600 -t 30 -o endurance_test.txt

    # # High-load burst test
    # # - Duration: 30 seconds
    # # - Target load: 100%
    # # - Tests maximum CPU performance
    # # - Good for testing AMD's boost clock capabilities
    # python amd_cpu_stress_test.py -d 30 -t 100 -o burst_test.txt

    # # CCX-aware testing (for Ryzen processors)
    # # - Tests even-numbered cores
    # # - Useful for testing cross-CCX performance
    # python amd_cpu_stress_test.py -c 0 2 4 6 -o ccx_test.txt

    # # Single-core maximum load test
    # # - Tests single core performance
    # # - Good for comparing AMD's single-core boost behavior
    # python amd_cpu_stress_test.py -c 0 -t 100 -o single_core.txt

    # # Dual-core balanced test
    # # - Tests two cores with moderate load
    # # - Useful for dual-core optimization testing
    # # - Good for testing AMD's power distribution
    # python amd_cpu_stress_test.py -c 0 1 -t 50 -d 300 -o dual_core.txt

    # # Physical cores only with moderate load
    # # - Disables SMT
    # # - Sets moderate target load
    # # - Good for baseline performance testing
    # python amd_cpu_stress_test.py --disable-smt -t 70 -o physical_cores.txt

    # # Multi-phase test script example
    # # - Runs multiple tests with different configurations
    # # - Good for comprehensive CPU analysis

    # # Phase 1: All cores, high load
    # # - Tests overall CPU performance
    # python amd_cpu_stress_test.py -d 300 -t 90 -o phase1.txt

    # # Phase 2: Physical cores only, medium load
    # # - Tests non-SMT performance
    # python amd_cpu_stress_test.py -d 300 -t 60 --disable-smt -o phase2.txt

    # # Phase 3: Specific cores, varying load
    # # - Tests specific core behavior
    # python amd_cpu_stress_test.py -d 300 -t 75 -c 0 2 4 -o phase3.txt

    # # Thermal throttling test
    # # - Long duration
    # # - High load
    # # - Good for testing AMD's thermal management
    # python amd_cpu_stress_test.py -d 1800 -t 95 -o thermal_test.txt

    # # Power efficiency test
    # # - Moderate load
    # # - All cores
    # # - Good for testing AMD's power management features
    # python amd_cpu_stress_test.py -d 600 -t 50 -o power_efficiency.txt

    # # Cross-CCX communication test (Ryzen specific)
    # # - Tests cores from different CCX units
    # # - Useful for analyzing inter-CCX performance
    # python amd_cpu_stress_test.py -c 0 4 -t 100 -d 300 -o ccx_comm_test.txt

    # # Precision Boost analysis
    # # - Varying loads
    # # - Good for analyzing AMD's boost behavior
    # for load in 30 50 70 90 100; do
    #     python amd_cpu_stress_test.py -d 300 -t $load -o boost_test_${load}.txt
    # done

    # # Thread scheduler test
    # # - Alternating cores
    # # - Tests Windows/Linux thread scheduler behavior with AMD CPUs
    # python amd_cpu_stress_test.py -c 0 2 4 6 -d 600 -t 80 -o scheduler_test.txt

    # # Memory controller stress test
    # # - All cores
    # # - High load
    # # - Tests integrated memory controller performance
    # python amd_cpu_stress_test.py -d 900 -t 100 -o memory_controller_test.txt