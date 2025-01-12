import os
import sys
import time
import argparse
import traceback
from datetime import datetime
from threading import Thread
from typing import List, Dict, Optional 

import psutil
import cupy as cp
import numpy as np
import pynvml
import copy 
from OpenGL.GL import *


def find_optimal_matrix_size(gpu_index: int, target_percent: float) -> tuple[int, dict]:
    """
    Find optimal matrix size and parameters that cause approximately target_percent GPU utilization
    using binary search approach
    
    Args:
        gpu_index (int): Index of the GPU to test
        target_percent (float): Target GPU usage percentage
        
    Returns:
        tuple[int, dict]: Optimal matrix size and corresponding parameters
    """
    def test_single_size(size: int, gpu_manager: Optional[GPUResourceManager] = None) -> tuple[float, dict]:
        """Test GPU utilization for a given matrix size and return utilization and best params"""
        print(f"\nTesting size {size}")
        try:
            # Initialize or reuse GPU manager
            if gpu_manager is None:
                gpu_manager = GPUResourceManager(size, device_id=gpu_index)
                
            # Force initialization of GPU resources
            gpu_manager.initialize()
            
            # Get device handle
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception as e:
                print(f"Error getting GPU handle: {e}")
                return 0, None
                
            # Test different parameter combinations
            best_util = 0
            best_params = None
            
            # Reduced parameter combinations for faster testing
            param_combinations = [
                {'sleep_time': 0.001, 'chunk_size': size // 10, 'num_streams': 4, 'sync_frequency': 2},
                {'sleep_time': 0.002, 'chunk_size': size // 8, 'num_streams': 4, 'sync_frequency': 2}
            ]
            
            # Warm-up run
            gpu_manager.perform_computation(param_combinations[0])
            time.sleep(0.2)
            
            for test_params in param_combinations:
                utils = []
                # Repetitions
                rep_times = 3
                for i in range(rep_times):
                    try:
                        gpu_manager.perform_computation(test_params)
                        readings = []
                        for _ in range(2):  # Readings
                            current_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                            readings.append(current_util)
                            time.sleep(0.1)
                        util = np.median(readings)
                        utils.append(util)
                        print(f"Params {test_params} - Run {i+1}: {util}%")
                    except Exception as e:
                        print(f"Error during computation {i+1}: {e}")
                        continue

                if utils:
                    avg_util = np.median(utils)  # Use median instead of mean
                    if abs(avg_util - target_percent) < abs(best_util - target_percent):
                        best_util = avg_util
                        best_params = test_params.copy()
                        print(f"New best params: {best_params} (util: {best_util:.1f}%)")

            return best_util, best_params
                
        except Exception as e:
            print(f"\nError testing size {size}: {e}")
            traceback.print_exc()
            return 0, None

    # Initialize NVML and get GPU info
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = info.total / (1024**2)  # Convert to MB
    except Exception as e:
        print(f"Error initializing NVML: {e}")
        return 5000, {'sleep_time': 0.001, 'chunk_size': 500, 'num_streams': 4, 'sync_frequency': 2}

    # Calculate maximum safe matrix size
    bytes_per_element = 4  # float32
    max_elements = (total_memory * 1024 * 1024 * 0.3) / (2 * bytes_per_element)
    max_matrix_size = int(np.sqrt(max_elements))
    
    # Binary search parameters
    left = 1000  # Minimum size
    right = min(10000, max_matrix_size)  # Maximum size
    best_size = 5000  # Default size
    best_diff = float('inf')
    best_util = 0
    best_params = {'sleep_time': 0.001, 'chunk_size': 500, 'num_streams': 4, 'sync_frequency': 2}
    gpu_manager = None
    
    try:
        # Binary search for optimal size
        while left <= right:
            size = (left + right) // 2
            print(f"\nTesting matrix size: {size} (range: {left}-{right})")
            
            # Create new GPU manager if needed
            if gpu_manager is None or gpu_manager.matrix_size != size:
                if gpu_manager is not None:
                    gpu_manager.cleanup()
                gpu_manager = GPUResourceManager(size, device_id=gpu_index)
            
            util, params = test_single_size(size, gpu_manager)
            
            if util > 0:
                diff = abs(util - target_percent)
                if diff < best_diff:
                    best_diff = diff
                    best_size = size
                    best_util = util
                    best_params = params
                    print(f"New best size: {best_size} (util: {best_util:.1f}%)")
                    
                    # Early termination if result is good enough
                    if diff <= 3.0:  # More aggressive threshold
                        print(f"Found good enough result (diff: {diff:.1f}%), stopping search.")
                        break
                
                # Adjust search range
                if util < target_percent:
                    left = size + 500  # Larger step size
                else:
                    right = size - 500
            else:
                right = size - 500
            
            # Break if range becomes too small
            if right - left < 500:
                break
    
    finally:
        if gpu_manager is not None:
            gpu_manager.cleanup()
    
    print(f"\nSelected size: {best_size}")
    print(f"Expected utilization: {best_util:.1f}%")
    print(f"Parameters: {best_params}")
    
    return best_size, best_params


class MultiVarPIDController:
    """Multi-variable PID Controller for GPU load management with fine-tuning mode"""
    def __init__(self, target_percent, matrix_size):
        # Control targets and limits
        self.target = target_percent
        self.matrix_size = matrix_size
        
        # Fine-tuning mode parameters
        self.fine_tuning = False
        self.fine_tuning_threshold = 3.0  # Enter fine-tuning when error < 3%
        self.stability_count = 0
        self.required_stable_readings = 10  # Readings needed to enter fine-tuning
        self.gain_reduction_factor = 0.2  # Reduce gains to 20% in fine-tuning mode
        
        # Initialize gains and store original values
        self._init_gains(target_percent)
        self.original_gains = copy.deepcopy(self.gains)
        
        # Initialize limits based on matrix size and target load
        self._init_limits(matrix_size, target_percent)
        
        # Add rate limiting
        self.max_change_rates = {
            'sleep': 0.001,  # max change per update
            'chunk': matrix_size // 100,  # max change per update
            'streams': 1,    # max change per update
            'sync': 1        # max change per update
        }
        
        # Initialize state variables
        self.state = {
            'sleep': {'integral': 0, 'last_error': 0, 'value': self._get_initial_sleep()},
            'chunk': {'integral': 0, 'last_error': 0, 'value': self._get_initial_chunk()},
            'streams': {'integral': 0, 'last_error': 0, 'value': self._get_initial_streams()},
            'sync': {'integral': 0, 'last_error': 0, 'value': self._get_initial_sync()}
        }
        
        # Previous output values for rate limiting
        self.prev_outputs = {
            'sleep': self._get_initial_sleep(),
            'chunk': self._get_initial_chunk(),
            'streams': self._get_initial_streams(),
            'sync': self._get_initial_sync()
        }
        
        self.last_time = time.time()
        self.integral_limit = 50.0  # Reduced integral limit

    def _init_gains(self, target_percent):
        """Initialize PID gains based on target load with more conservative values"""
        if target_percent < 30:  # Low load
            self.gains = {
                'sleep': {'kp': 0.008, 'ki': 0.0008, 'kd': 0.00008},
                'chunk': {'kp': 0.08, 'ki': 0.008, 'kd': 0.0008},
                'streams': {'kp': 0.02, 'ki': 0.002, 'kd': 0.0002},
                'sync': {'kp': 0.06, 'ki': 0.006, 'kd': 0.0006}
            }
        elif target_percent < 70:  # Medium load - much more conservative
            self.gains = {
                'sleep': {'kp': 0.005, 'ki': 0.0005, 'kd': 0.00005},
                'chunk': {'kp': 0.05, 'ki': 0.005, 'kd': 0.0005},
                'streams': {'kp': 0.02, 'ki': 0.002, 'kd': 0.0002},
                'sync': {'kp': 0.04, 'ki': 0.004, 'kd': 0.0004}
            }
        else:  # High load
            self.gains = {
                'sleep': {'kp': 0.003, 'ki': 0.0003, 'kd': 0.00003},
                'chunk': {'kp': 0.03, 'ki': 0.003, 'kd': 0.0003},
                'streams': {'kp': 0.04, 'ki': 0.004, 'kd': 0.0004},
                'sync': {'kp': 0.02, 'ki': 0.002, 'kd': 0.0002}
            }

    def _init_limits(self, matrix_size, target_percent):
        """Initialize control variable limits based on matrix size and target load"""
        # Base limits
        self.limits = {
            'sleep': (0.0001, 0.01),
            'chunk': (max(50, matrix_size // 100), max(200, matrix_size // 4)),
            'streams': (2, 16),
            'sync': (1, 10)
        }
        
        # Adjust limits based on target load
        if target_percent < 30:
            self.limits['sleep'] = (0.0001, 0.005)
            self.limits['chunk'] = (max(50, matrix_size // 100), max(100, matrix_size // 8))
            self.limits['streams'] = (2, 8)
            self.limits['sync'] = (1, 4)
        elif target_percent > 70:
            self.limits['sleep'] = (0.0001, 0.003)
            self.limits['chunk'] = (max(100, matrix_size // 50), max(400, matrix_size // 2))
            self.limits['streams'] = (4, 16)
            self.limits['sync'] = (2, 10)

    def _get_initial_sleep(self):
        """More conservative initial sleep times"""
        if self.target < 30:
            return 0.003
        elif self.target < 70:
            return 0.002
        else:
            return 0.001

    def _get_initial_chunk(self):
        """More conservative initial chunk sizes"""
        if self.target < 30:
            return max(50, self.matrix_size // 40)
        elif self.target < 70:
            return max(100, self.matrix_size // 20)
        else:
            return max(200, self.matrix_size // 10)

    def _get_initial_streams(self):
        """Get initial number of streams based on target load"""
        if self.target < 30:
            return 2
        elif self.target < 70:
            return 4
        else:
            return 8

    def _get_initial_sync(self):
        """Get initial sync frequency based on target load"""
        if self.target < 30:
            return 1
        elif self.target < 70:
            return 2
        else:
            return 4

    def _reduce_gains_for_fine_tuning(self):
        """Reduce PID gains for fine-tuning mode"""
        for var_name in self.gains:
            for gain_type in self.gains[var_name]:
                self.gains[var_name][gain_type] = (
                    self.original_gains[var_name][gain_type] * self.gain_reduction_factor
                )
        print("Entering fine-tuning mode with reduced gains")

    def _restore_original_gains(self):
        """Restore original PID gains"""
        self.gains = copy.deepcopy(self.original_gains)
        print("Exiting fine-tuning mode, restored original gains")

    def compute(self, current_util, load_stability):
        """
        Compute new control values with fine-tuning mode support
        
        Args:
            current_util (float): Current GPU utilization
            load_stability (float): Measure of load stability (0-1)
            
        Returns:
            dict: Updated control parameters
        """
        error = self.target - current_util  # Changed from abs() to maintain sign
        
        # Check if we should enter or exit fine-tuning mode
        if not self.fine_tuning:
            if abs(error) <= self.fine_tuning_threshold:
                self.stability_count += 1
                if self.stability_count >= self.required_stable_readings:
                    self.fine_tuning = True
                    self._reduce_gains_for_fine_tuning()
            else:
                self.stability_count = 0
        else:
            # Exit fine-tuning if error becomes too large
            if abs(error) > self.fine_tuning_threshold * 1.5:  # Add some hysteresis
                self.fine_tuning = False
                self.stability_count = 0
                self._restore_original_gains()

        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            return self._get_current_values()

        # Update each control variable
        for var_name in self.state:
            state = self.state[var_name]
            gains = self.gains[var_name]
            limits = self.limits[var_name]
            
            # Update integral with anti-windup
            state['integral'] = max(min(
                state['integral'] + error * dt,
                self.integral_limit if not self.fine_tuning else self.integral_limit * 0.3
            ), -self.integral_limit)
            
            # Calculate PID terms
            p_term = gains['kp'] * error
            i_term = gains['ki'] * state['integral']
            d_term = gains['kd'] * (error - state['last_error']) / dt if dt > 0 else 0
            
            # Calculate raw output with variable-specific strategies
            output = state['value']
            if self.fine_tuning:
                # In fine-tuning mode, make smaller adjustments
                if var_name == 'sleep':
                    output += (p_term + i_term + d_term) * 0.5
                elif var_name == 'chunk':
                    output += (p_term + i_term + d_term) * 0.3 * (1 - load_stability)
                elif var_name == 'streams':
                    output += (p_term + i_term + d_term) * 0.2 * abs(error) / 100
                elif var_name == 'sync':
                    output += (p_term + i_term + d_term) * 0.4 * load_stability
            else:
                # Normal mode adjustments
                if var_name == 'sleep':
                    output += (p_term + i_term + d_term)
                elif var_name == 'chunk':
                    output += (p_term + i_term + d_term) * (1 - load_stability)
                elif var_name == 'streams':
                    output += (p_term + i_term + d_term) * abs(error) / 100
                elif var_name == 'sync':
                    output += (p_term + i_term + d_term) * load_stability
            
            # Apply rate limiting
            max_change = self.max_change_rates[var_name]
            prev_output = self.prev_outputs[var_name]
            output = max(prev_output - max_change, 
                        min(prev_output + max_change, output))
            
            # Store for next iteration
            self.prev_outputs[var_name] = output
            
            # Apply limits with tighter bounds in fine-tuning mode
            if self.fine_tuning:
                # Calculate tighter limits around current value
                current = state['value']
                limit_range = limits[1] - limits[0]
                fine_tune_limits = (
                    max(limits[0], current - limit_range * 0.05),  # Reduced to 5%
                    min(limits[1], current + limit_range * 0.05)
                )
                state['value'] = max(fine_tune_limits[0], min(fine_tune_limits[1], output))
            else:
                state['value'] = max(limits[0], min(limits[1], output))
            
            state['last_error'] = error
        
        self.last_time = current_time
        return self._get_current_values()

    def _get_current_values(self):
        """Get current values of all control variables"""
        return {
            'sleep_time': self.state['sleep']['value'],
            'chunk_size': int(self.state['chunk']['value']),
            'num_streams': int(self.state['streams']['value']),
            'sync_frequency': int(self.state['sync']['value'])
        }

    def get_status(self):
        """Get controller status information"""
        return {
            'fine_tuning': self.fine_tuning,
            'stability_count': self.stability_count,
            'current_values': self._get_current_values()
        }

    def reset(self):
        """Reset controller state"""
        self.fine_tuning = False
        self.stability_count = 0
        self.gains = copy.deepcopy(self.original_gains)
        for state in self.state.values():
            state['integral'] = 0
            state['last_error'] = 0

class GPUResourceManager:
    """Manages GPU resources to maintain stable GPU utilization"""
    def __init__(self, matrix_size, num_streams=4, device_id=0):
        self.matrix_size = matrix_size
        self.device_id = device_id
        self.num_streams = num_streams
        
        # Resource flags
        self.initialized = False
        self.resources_held = False
        self.last_use_time = 0
        self.hold_duration = 5  # Hold resources for 5 seconds
        
        # Resource containers
        self.streams = None
        self.matrix1 = None
        self.matrix2 = None
        self.result_buffer = None
        
    def initialize(self):
        """Initialize GPU resources"""
        if self.initialized:
            return
            
        with cp.cuda.Device(self.device_id):
            # Create persistent streams
            self.streams = [cp.cuda.Stream(non_blocking=True) for _ in range(self.num_streams)]
            
            # Allocate matrices with pinned memory
            self.matrix1 = cp.random.rand(self.matrix_size, self.matrix_size, dtype=cp.float32)
            self.matrix2 = cp.random.rand(self.matrix_size, self.matrix_size, dtype=cp.float32)
            
            # Pre-allocate result buffer
            self.result_buffer = cp.zeros((self.matrix_size, self.matrix_size), dtype=cp.float32)
            
            # Force initialization
            cp.cuda.Stream.null.synchronize()
            
        self.initialized = True
        self.resources_held = True
        self.last_use_time = time.time()
        
    def release_if_idle(self):
        """Release resources if they've been idle for too long"""
        if not self.resources_held:
            return
            
        current_time = time.time()
        if current_time - self.last_use_time > self.hold_duration:
            self._release_resources()
            
    def _release_resources(self):
        """Release GPU resources"""
        if not self.resources_held:
            return
            
        with cp.cuda.Device(self.device_id):
            # Synchronize all streams before release
            for stream in self.streams:
                stream.synchronize()
                
            # Release memory
            self.matrix1 = None
            self.matrix2 = None
            self.result_buffer = None
            
            # Clear memory pool
            cp.get_default_memory_pool().free_all_blocks()
            
        self.resources_held = False
        
    def perform_computation(self, params):
        """Perform GPU computation while maintaining resources"""
        if not self.initialized:
            self.initialize()
            
        with cp.cuda.Device(self.device_id):
            stream_idx = 0
            chunk_count = 0
            
            for i in range(0, self.matrix_size, params['chunk_size']):
                end_idx = min(i + params['chunk_size'], self.matrix_size)
                
                with self.streams[stream_idx]:
                    # Reuse pre-allocated result buffer
                    cp.dot(
                        self.matrix1[i:end_idx], 
                        self.matrix2, 
                        out=self.result_buffer[i:end_idx]
                    )
                    
                    chunk_count += 1
                    if chunk_count % params['sync_frequency'] == 0:
                        self.streams[stream_idx].synchronize()
                        if params['sleep_time'] > 0:
                            time.sleep(params['sleep_time'])
                
                stream_idx = (stream_idx + 1) % self.num_streams
            
            # Update last use time
            self.last_use_time = time.time()
            
    def cleanup(self):
        """Final cleanup"""
        self._release_resources()
        self.initialized = False


class LoadSmoothingController:
    """
    A wrapper controller that smooths the PID controller's output and applies penalties
    when the system stabilizes at wrong levels.
    """
    def __init__(self, pid_controller, config=None):
        """Initialize the controller with PID controller and optional config"""
        self.pid_controller = pid_controller
        self.config = {
            'max_util_rate': 5.0,
            'smoothing_factor': 0.7,
            'history_size': 5,
            'min_damping': 0.3,
            'recovery_rate': 1.1,
            'error_threshold': 10.0,
            'penalty_window': 5,
            'penalty_factor': 0.5,
            'penalty_recovery': 0.95,
        }
        if config:
            self.config.update(config)
            
        self.util_history = []
        self.error_history = []
        self.last_output = None
        self.current_damping = 1.0
        self.penalty_active = False
        self.penalty_multiplier = 1.0
    
    def update_config(self, new_config):
        """Update controller configuration"""
        if not isinstance(new_config, dict):
            raise ValueError("new_config must be a dictionary")
        
        # Validate required keys
        required_keys = {
            'max_util_rate', 'smoothing_factor', 'history_size',
            'min_damping', 'recovery_rate', 'error_threshold',
            'penalty_window', 'penalty_factor', 'penalty_recovery'
        }
        
        missing_keys = required_keys - set(new_config.keys())
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
            
        # Validate values
        if new_config['smoothing_factor'] <= 0 or new_config['smoothing_factor'] >= 1:
            raise ValueError("smoothing_factor must be between 0 and 1")
        if new_config['min_damping'] <= 0 or new_config['min_damping'] > 1:
            raise ValueError("min_damping must be between 0 and 1")
        if new_config['max_util_rate'] <= 0:
            raise ValueError("max_util_rate must be positive")
        
        # Update configuration
        self.config.update(new_config)
        
        # Reset damping when config changes to prevent instability
        self.current_damping = 1.0

    def _update_history(self, current_util, current_time):
        """Update utilization history and maintain its size"""
        self.util_history.append((current_time, current_util))
        while len(self.util_history) > self.config['history_size']:
            self.util_history.pop(0)
            
    def _calculate_change_rate(self):
        """Calculate current rate of change in utilization (%/s)"""
        if len(self.util_history) < 2:
            return 0.0
            
        recent_time, recent_util = self.util_history[-1]
        old_time, old_util = self.util_history[0]
        
        time_diff = recent_time - old_time
        if time_diff <= 0:
            return 0.0
            
        return abs(recent_util - old_util) / time_diff

    def _smooth_output(self, new_output):
        """Apply exponential smoothing to the output values"""
        if self.last_output is None:
            self.last_output = new_output
            return new_output
            
        smoothed = {}
        for key in new_output:
            if key in self.last_output:
                # Apply smoothing and ensure integer values where needed
                smoothed_value = (self.config['smoothing_factor'] * new_output[key] +
                               (1 - self.config['smoothing_factor']) * self.last_output[key])
                # Ensure integer values for specific parameters
                if key in ['chunk_size', 'num_streams', 'sync_frequency']:
                    smoothed_value = int(round(smoothed_value))
                smoothed[key] = smoothed_value
            else:
                smoothed[key] = new_output[key]
                
        self.last_output = smoothed
        return smoothed

    def _apply_damping(self, output, damping):
        """Apply damping factor to control outputs"""
        if self.last_output is None:
            return output
            
        damped = {}
        for key in output:
            if key in self.last_output:
                # Calculate damped value between current output and last output
                damped_value = (damping * output[key] +
                             (1 - damping) * self.last_output[key])
                # Ensure integer values for specific parameters
                if key in ['chunk_size', 'num_streams', 'sync_frequency']:
                    damped_value = int(round(damped_value))
                damped[key] = damped_value
            else:
                damped[key] = output[key]
                
        return damped

    def _update_penalty_state(self, current_util):
        """
        Update penalty state based on current error, excluding outliers
        """
        target = self.pid_controller.target
        current_error = abs(current_util - target)
        self.error_history.append(current_error)
        
        # Keep error history within window
        while len(self.error_history) > self.config['penalty_window']:
            self.error_history.pop(0)
            
        # Check if we have enough samples
        if len(self.error_history) >= self.config['penalty_window']:
            # Calculate quartiles to identify outliers
            errors = np.array(self.error_history)
            q1 = np.percentile(errors, 25)
            q3 = np.percentile(errors, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filter out outliers
            filtered_errors = errors[(errors >= lower_bound) & (errors <= upper_bound)]
            
            # Only proceed if we have enough non-outlier samples
            if len(filtered_errors) >= max(3, self.config['penalty_window'] // 2):
                avg_error = np.mean(filtered_errors)
                
                if avg_error > self.config['error_threshold']:
                    if not self.penalty_active:
                        print(f"Activating penalty - Average error (excluding outliers): {avg_error:.1f}%")
                    self.penalty_active = True
                    # Increase penalty strength
                    self.penalty_multiplier = max(
                        self.config['penalty_factor'],
                        self.penalty_multiplier * self.config['penalty_factor']
                    )
                else:
                    if self.penalty_active:
                        # Gradually recover from penalty
                        self.penalty_multiplier = min(
                            1.0,
                            self.penalty_multiplier / self.config['penalty_recovery']
                        )
                        if self.penalty_multiplier > 0.95:  # Almost recovered
                            self.penalty_active = False
                            print("Deactivating penalty - Error within threshold")
        
    def _adjust_damping(self, change_rate):
        """Adjust damping factor based on change rate and penalty"""
        if change_rate > self.config['max_util_rate']:
            # Increase damping (reduce control action) when change rate is too high
            target_damping = max(
                self.config['min_damping'],
                self.config['max_util_rate'] / change_rate
            )
            self.current_damping = min(self.current_damping, target_damping)
        else:
            # Gradually recover damping when change rate is acceptable
            self.current_damping = min(
                1.0,
                self.current_damping * self.config['recovery_rate']
            )
        
        # Apply penalty by reducing damping if active
        if self.penalty_active:
            self.current_damping *= self.penalty_multiplier
            
        return self.current_damping
        
    def compute(self, current_util, load_stability):
        """
        Compute smoothed control values with penalty consideration
        """
        current_time = time.time()
        
        # Update history and calculate change rate
        self._update_history(current_util, current_time)
        change_rate = self._calculate_change_rate()
        
        # Update penalty state
        self._update_penalty_state(current_util)
        
        # Get raw PID output
        pid_output = self.pid_controller.compute(current_util, load_stability)
        
        # Apply smoothing and damping
        if change_rate > 0:
            # Calculate and apply damping based on change rate and penalty
            damping = self._adjust_damping(change_rate)
            output = self._apply_damping(pid_output, damping)
            
            # Apply additional exponential smoothing
            output = self._smooth_output(output)
        else:
            output = pid_output
            
        return output
        
    def get_status(self):
        """Get controller status including penalty metrics"""
        status = self.pid_controller.get_status()
        status.update({
            'change_rate': self._calculate_change_rate(),
            'current_damping': self.current_damping,
            'penalty_active': self.penalty_active,
            'penalty_multiplier': self.penalty_multiplier,
            'avg_error': sum(self.error_history) / len(self.error_history) if self.error_history else 0.0
        })
        return status
        
    def reset(self):
        """Reset controller state including penalty state"""
        self.pid_controller.reset()
        self.util_history.clear()
        self.error_history.clear()
        self.last_output = None
        self.current_damping = 1.0
        self.penalty_active = False
        self.penalty_multiplier = 1.0

def calculate_stability(current_util, util_history=None, history_size=10):
    """Calculate load stability based on utilization history"""
    if util_history is None:
        util_history = []
        
    util_history.append(current_util)
    if len(util_history) > history_size:
        util_history.pop(0)
        
    if len(util_history) >= 2:
        variations = np.diff(util_history)
        stability = 1.0 / (1.0 + np.std(variations))
    else:
        stability = 0.5
        
    return stability, util_history


def single_matrix_stress(stop_flag, target_percent, gpu_index, matrix_size_or_tuple, initial_params=None):
    """
    Function to perform GPU stress test with controlled usage
    
    Args:
        stop_flag (list): Flag to control the task execution
        target_percent (float): Target GPU usage percentage (1-100)
        gpu_index (int): Index of the GPU to stress test
        matrix_size_or_tuple: Either an int or a tuple(int, dict) from find_optimal_matrix_size
        initial_params (dict, optional): Initial parameters from optimization
    """
    def ensure_integer_params(params):
        """Ensure certain parameters are integers and within valid ranges"""
        result = params.copy()
        
        # Define minimum and maximum values
        limits = {
            'chunk_size': (1, 10000),
            'num_streams': (1, 32),
            'sync_frequency': (1, 10),
            'sleep_time': (0.0001, 0.1)
        }
        
        for key, (min_val, max_val) in limits.items():
            if key in result:
                if key == 'sleep_time':
                    result[key] = max(min_val, min(float(result[key]), max_val))
                else:
                    result[key] = max(min_val, min(int(round(float(result[key]))), max_val))
        
        return result

    def get_gpu_utilization():
        """Safely get GPU utilization with retries"""
        max_retries = 3
        for _ in range(max_retries):
            try:
                return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            except pynvml.NVMLError:
                time.sleep(0.1)
                continue
        return None

    def get_gpu_temperature():
        """Safely get GPU temperature"""
        try:
            return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError:
            return None

    # Handle matrix_size input
    if isinstance(matrix_size_or_tuple, tuple):
        matrix_size, initial_params = matrix_size_or_tuple
    else:
        matrix_size = matrix_size_or_tuple
        if initial_params is None:
            matrix_size, initial_params = find_optimal_matrix_size(gpu_index, target_percent)

    if initial_params is None:
        raise ValueError("No valid initial parameters available")

    initial_params = ensure_integer_params(initial_params)
    params = initial_params.copy()

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    except pynvml.NVMLError as e:
        raise RuntimeError(f"Failed to initialize NVML: {str(e)}")

    pid_controller = MultiVarPIDController(target_percent, matrix_size)
    smoother = LoadSmoothingController(pid_controller)
    
    gpu_manager = GPUResourceManager(matrix_size, device_id=gpu_index)
    gpu_manager.initialize()
    
    util_history = []
    
    # Control parameters
    if target_percent <= 30:
        stability_threshold = 3.0
        max_adjustment = 0.2
    elif target_percent >= 70:
        stability_threshold = 5.0
        max_adjustment = 0.3
    else:
        stability_threshold = 4.0
        max_adjustment = 0.25

    try:
        # Warm-up phase
        warm_up_utils = []
        warm_up_success = False
        
        # Ensure initial parameters are valid
        params = ensure_integer_params(initial_params.copy())
        last_successful_params = params.copy()
        
        # Initial stabilization
        for _ in range(10):
            gpu_manager.perform_computation(params)
            current_util = get_gpu_utilization()
            
            if current_util is not None:
                warm_up_utils.append(current_util)
                
                if len(warm_up_utils) >= 3:
                    avg_util = np.mean(warm_up_utils[-3:])
                    error = target_percent - avg_util
                    
                    if abs(error) <= stability_threshold:
                        warm_up_success = True
                        break
                    elif len(warm_up_utils) >= 3:
                        # Gradual adjustment
                        if abs(error) > 20:
                            adjustment = 0.2 * (error / target_percent)
                        else:
                            adjustment = 0.1 * (error / target_percent)
                        
                        adjustment = max(min(adjustment, max_adjustment), -max_adjustment)
                        
                        new_params = params.copy()
                        for key in ['chunk_size', 'num_streams']:
                            if key in new_params:
                                current_value = new_params[key]
                                new_value = int(current_value * (1 + adjustment))
                                new_params[key] = new_value
                        
                        params = ensure_integer_params(new_params)
                        warm_up_utils = warm_up_utils[-2:]
            
            if warm_up_success:
                break
        
        last_successful_params = params.copy()
        
        # Main control loop
        while not stop_flag[0]:
            try:
                gpu_manager.perform_computation(params)
                current_util = get_gpu_utilization()
                
                if current_util is None:
                    params = last_successful_params.copy()
                    continue
                
                error = target_percent - current_util
                
                # Update stability tracking
                stability, util_history = calculate_stability(current_util, util_history)
                
                if abs(error) <= stability_threshold:
                    last_successful_params = params.copy()
                    continue
                
                # Adjust parameters based on error
                if abs(error) > 10:
                    adjustment = 0.15 * (error / target_percent)
                else:
                    adjustment = 0.05 * (error / target_percent)
                
                # Apply temperature factor
                temp = get_gpu_temperature()
                if temp is not None:
                    temp_factor = 1.0 + max(0, (temp - 40) / 60)  # Increase adjustment for higher temps
                    adjustment *= temp_factor
                
                adjustment = max(min(adjustment, max_adjustment), -max_adjustment)
                
                new_params = params.copy()
                for key in ['chunk_size', 'num_streams']:
                    if key in new_params:
                        current_value = new_params[key]
                        new_value = int(current_value * (1 + adjustment))
                        new_params[key] = new_value
                
                params = ensure_integer_params(new_params)
                
            except Exception as e:
                params = last_successful_params.copy()
                continue
            
    except KeyboardInterrupt:
        pass
    finally:
        try:
            gpu_manager.cleanup()
        except:
            pass
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def single_matrix_stress_test(gpu_index: int, duration: int, target_percent: float, 
                   matrix_size: int, shared_stop_flag: Optional[list] = None) -> dict:
    """
    Test a single GPU and return its metrics
    
    Args:
        gpu_index (int): Index of the GPU to test
        duration (int): Test duration in seconds
        target_percent (float): Target GPU usage percentage
        matrix_size (int): Size of the computation matrices
        shared_stop_flag (list, optional): Shared stop flag for parallel testing
    
    Returns:
        dict: Dictionary containing test results and metrics
    """
    # Initialize metrics storage
    metrics = {
        'timestamps': [],
        'utilizations': [],
        'memory_usage': [], 
        'temperatures': [],
        'frequencies': [],
        'power_values': [],
        'battery_stats': []
    }

    # Use provided stop flag or create new one
    stop_flag = shared_stop_flag if shared_stop_flag is not None else [False]

    # Initialize NVML and get handle
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    # Find optimal parameters first
    print(f"\nFinding optimal parameters for GPU {gpu_index}...")
    _, initial_params = find_optimal_matrix_size(gpu_index, target_percent)

    # Start stress task in a separate thread
    stress_thread = Thread(target=single_matrix_stress, 
                         args=(stop_flag, target_percent, gpu_index, matrix_size, initial_params))
    stress_thread.start()

    # Parameters for stabilization check
    stabilization_params = {
        'tolerance': 5.0,
        'required_readings': 5,
        'check_interval': 0.2,
        'max_wait_time': 30
    }

    # Wait for GPU usage to stabilize
    print(f"Waiting for GPU {gpu_index} to stabilize at {target_percent}%...")
    stabilization_count = 0
    start_stabilization_time = time.time()
    
    while not (shared_stop_flag and shared_stop_flag[0]):
        try:
            # Get current GPU utilization
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            deviation = abs(gpu_util - target_percent)
            
            # Check if within tolerance
            if deviation <= stabilization_params['tolerance']:
                stabilization_count += 1
                status = "Using optimal parameters"
            else:
                stabilization_count = 0
                status = "Adjusting to target"
            
            # Calculate and display progress
            progress = min(stabilization_count / stabilization_params['required_readings'], 1.0)
            bar_length = 20
            filled_length = int(bar_length * progress)
            bar = "=" * filled_length + "-" * (bar_length - filled_length)
            
            # Only show progress when getting close to target
            if gpu_util > target_percent * 0.5:
                print(f"\rGPU {gpu_index} [{bar}] {gpu_util:.1f}% ({status}) "
                      f"Target: {target_percent}% ±{stabilization_params['tolerance']}%", end="")
            
            # Check stabilization conditions
            if stabilization_count >= stabilization_params['required_readings']:
                print(f"\nGPU {gpu_index} stabilized within {stabilization_params['tolerance']}% of target.")
                break
            
            # Check timeout
            if time.time() - start_stabilization_time > stabilization_params['max_wait_time']:
                print(f"\nWarning: GPU {gpu_index} failed to fully stabilize within "
                      f"{stabilization_params['max_wait_time']}s.")
                print(f"Continuing with current utilization ({gpu_util:.1f}%).")
                break
            
            time.sleep(stabilization_params['check_interval'])
            
        except pynvml.NVMLError as e:
            print(f"\nError reading GPU {gpu_index} utilization: {e}")
            break

    # Main monitoring loop
    start_time = time.time()
    try:
        while (time.time() - start_time) < duration and not (shared_stop_flag and shared_stop_flag[0]):
            try:
                # Collect current metrics
                current_metrics = collect_gpu_metrics(handle)
                update_metrics_storage(metrics, current_metrics)

                # Calculate deviation and status
                deviation = abs(current_metrics['utilization'] - target_percent)
                status = "Successfully maintain" if deviation <= stabilization_params['tolerance'] else "Unstable issue"

                # Display current status
                remaining = duration - (time.time() - start_time)
                print_status_line(gpu_index, current_metrics, target_percent, 
                                stabilization_params['tolerance'], remaining, status)

                time.sleep(0.5)

            except pynvml.NVMLError as e:
                print(f"\nError reading GPU {gpu_index} metrics: {e}")
                break

    except KeyboardInterrupt:
        print(f"\nTest interrupted for GPU {gpu_index}")
    finally:
        if shared_stop_flag is None:
            stop_flag[0] = True
        stress_thread.join()

    # Print summary only if we have collected data
    print("\n\nTest Summary:")
    if len(metrics['utilizations']) > 0:
        print(f"Average GPU Utilization: {np.mean(metrics['utilizations']):.1f}%")
        print(f"Average GPU Temperature: {np.mean(metrics['temperatures']):.1f}°C")
        print(f"Average GPU Frequency: {np.mean(metrics['frequencies']):.1f}MHz")

        try:
            max_graphics_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            print(f"Maximum GPU Frequency: {max_graphics_clock}MHz")
            print(f"Average/Max GPU Frequency Ratio: {(np.mean(metrics['frequencies'])/max_graphics_clock):.1%}")
        except pynvml.NVMLError as e:
            print(f"Unable to get maximum GPU frequency: {e}")
        
        if len(metrics['power_values']) > 0:
            print(f"Average GPU Power: {np.mean(metrics['power_values']):.1f}W")
            try:
                max_power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
                print(f"GPU Power Limit: {max_power_limit:.1f}W")
                print(f"Average/Max Power Ratio: {(np.mean(metrics['power_values'])/max_power_limit):.1%}")
            except pynvml.NVMLError as e:
                print(f"Unable to get GPU power limit: {e}")

        # Calculate throttling score
        try:
            gamma = 1 - np.array(metrics['frequencies']) / max_graphics_clock * (1 - np.array(metrics['utilizations']) / 100)
            score_gamma = np.mean(np.exp(- gamma))
            print(f"Score_gamma: {score_gamma:.5f}, gamma: {np.mean(gamma):.5f}")
            print(f"If Score_gamma approaches 1 (gamma approaches 0), GPU {gpu_index} throttling is insignificant")
            print(f"If Score_gamma approaches 0 (gamma approaches 1), GPU {gpu_index} throttling is significant")
        except Exception as e:
            print(f"Unable to calculate throttling score: {e}")
    else:
        print("No data was collected during the test.")

    return metrics


def collect_gpu_metrics(handle):
    """Collect current GPU metrics"""
    current_time = time.strftime("%H:%M:%S")
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    gpu_freq = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
    
    try:
        gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert mW to W
    except pynvml.NVMLError:
        gpu_power = 0

    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    memory_usage = (memory_info.used / memory_info.total) * 100

    battery = psutil.sensors_battery()
    battery_stat = ("Charging" if battery.power_plugged else f"Battery: {battery.percent}%") if battery else "No battery"

    return {
        'time': current_time,
        'utilization': gpu_util,
        'memory_usage': memory_usage,
        'temperature': gpu_temp,
        'frequency': gpu_freq,
        'power': gpu_power,
        'battery': battery_stat
    }


def update_metrics_storage(metrics, current_metrics):
    """Update metrics storage with current values"""
    metrics['timestamps'].append(current_metrics['time'])
    metrics['utilizations'].append(current_metrics['utilization'])
    metrics['memory_usage'].append(current_metrics['memory_usage'])
    metrics['temperatures'].append(current_metrics['temperature'])
    metrics['frequencies'].append(current_metrics['frequency'])
    if current_metrics['power'] > 0:
        metrics['power_values'].append(current_metrics['power'])
    metrics['battery_stats'].append(current_metrics['battery'])


def print_status_line(gpu_index, metrics, target, tolerance, remaining, status):
    """Print current status information"""
    print(f"\rGPU {gpu_index} - Util: {metrics['utilization']}% ({status}) | "
          f"Target: {target}% ±{tolerance}% | "
          f"Temp: {metrics['temperature']}°C | "
          f"Freq: {metrics['frequency']}MHz | "
          f"Power: {metrics['power']}W | "
          f"Time remaining: {remaining:.1f}s | "
          f"{metrics['battery']}", end="")


class Logger:
    """Logger class to handle both console and file output"""
    def __init__(self, filepath=None):
        self.terminal = sys.stdout
        self.filepath = filepath
        self.log_file = None
        if filepath:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.log_file = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        """Write message to both terminal and log file"""
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        """Flush both outputs"""
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()

    def close(self):
        """Close log file if it exists"""
        if self.log_file:
            self.log_file.close()


def matrix_stress(duration: int, target_percent: float, gpu_index: int = 0, log_file: str = None) -> Dict[int, dict]:
    """
    Run GPU matrix stress test on specified GPU
    
    Args:
        duration (int): Test duration in seconds
        target_percent (float): Target GPU usage percentage
        gpu_index (int): Index of the GPU to test
        log_file (str): Path to log file
    
    Returns:
        Dict[int, dict]: Dictionary mapping GPU index to test metrics
    """
    # Setup logging
    original_stdout = sys.stdout
    logger = Logger(log_file) if log_file else None
    if logger:
        sys.stdout = logger

    try:
        # Initialize NVML
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        # Validate GPU index
        if gpu_index >= device_count:
            raise ValueError(f"Invalid GPU index {gpu_index}. System has {device_count} GPUs")

        # Print test configuration
        print("\nGPU Matrix Stress Test Configuration:")
        print(f"Duration: {duration} seconds")
        print(f"GPU {gpu_index}: Target {target_percent}%")
        print("\nInitializing test...\n")

        # Find optimal matrix size for GPU
        matrix_size = find_optimal_matrix_size(gpu_index, target_percent)

        # Run test and collect results
        results = {}
        results[gpu_index] = single_matrix_stress_test(gpu_index, duration, target_percent, matrix_size)

        return results

    except Exception as e:
        print(f"Error during stress test: {e}")
        return {}
    finally:
        # Cleanup
        if logger:
            sys.stdout = original_stdout
            logger.close()
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def simple_stress(stop_flag: list, target_percent: float, gpu_index: int, duration: int) -> dict:
    """
    Simple GPU stress test using basic arithmetic operations with continuous load adjustment
    
    Args:
        stop_flag (list): Flag to control the task execution
        target_percent (float): Target GPU usage percentage (1-100)
        gpu_index (int): Index of the GPU to stress test
        duration (int): Test duration in seconds
    
    Returns:
        dict: Dictionary containing test metrics
    """
    # Initialize metrics storage
    metrics = {
        'timestamps': [],
        'utilizations': [],
        'memory_usage': [], 
        'temperatures': [],
        'frequencies': [],
        'power_values': [],
        'battery_stats': []
    }

    try:
        # Initialize NVML and get device handle
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    except pynvml.NVMLError as e:
        raise RuntimeError(f"Failed to initialize NVML: {str(e)}")

    # Initialize GPU test object
    gpu_test = SimpleGPUStressTest(gpu_index)
    
    # Initialize current GPU utilization
    try:
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    except pynvml.NVMLError:
        gpu_util = 0
    
    print(f"Waiting for GPU {gpu_index} to stabilize at {target_percent}%...")
    
    # Stabilization phase variables
    start_time = None
    stabilization_count = 0  # Count consecutive stable readings
    STABILITY_THRESHOLD = 5  # Number of consecutive stable readings required
    STABILITY_MARGIN = 5     # Acceptable deviation from target (±5%)
    
    # Load adjustment variables
    last_adjustment_time = time.time()
    adjustment_interval = 0.1
    
    while not stop_flag[0]:
        try:
            current_time = time.time()
            
            # Check if it's time to adjust the load
            should_adjust = (current_time - last_adjustment_time) >= adjustment_interval
            
            # Perform GPU computation with current utilization feedback
            if should_adjust:
                gpu_test.perform_computation(gpu_util, target_percent)
                last_adjustment_time = current_time
            else:
                # Use current workload without adjustment
                gpu_test.perform_computation_fixed()
            
            try:
                # Get GPU metrics
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_usage = (memory_info.used / memory_info.total) * 100
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_freq = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                
                try:
                    gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except pynvml.NVMLError:
                    gpu_power = 0
                
                # Get battery status using psutil
                battery = psutil.sensors_battery()
                battery_stat = ("Charging" if battery.power_plugged else f"Battery: {battery.percent}%") if battery else "No battery"

                # Check if GPU has stabilized at target
                is_stable = abs(gpu_util - target_percent) <= STABILITY_MARGIN
                
                if not start_time:
                    # Still in stabilization phase
                    if is_stable:
                        stabilization_count += 1
                    else:
                        stabilization_count = 0

                    # Start timing only after stabilization
                    if stabilization_count >= STABILITY_THRESHOLD:
                        start_time = time.time()
                        print(f"\nGPU {gpu_index} has stabilized. Starting {duration} second test...")
                        # Initialize fresh metrics dictionary instead of clearing
                        metrics = {
                            'timestamps': [],
                            'utilizations': [],
                            'memory_usage': [], 
                            'temperatures': [],
                            'frequencies': [],
                            'power_values': [],
                            'battery_stats': []
                        }
                
                # Calculate remaining time only after stabilization
                if start_time is not None:
                    elapsed = int(current_time - start_time)
                    remaining = max(0, duration - elapsed)
                    
                    # Update metrics storage only during the test period
                    metrics['timestamps'].append(current_time)
                    metrics['utilizations'].append(gpu_util)
                    metrics['memory_usage'].append(memory_usage)
                    metrics['temperatures'].append(gpu_temp)
                    metrics['frequencies'].append(gpu_freq)
                    metrics['power_values'].append(gpu_power)
                    metrics['battery_stats'].append(battery_stat)
                else:
                    remaining = duration

                # Determine GPU stability status
                status = "Successfully maintain" if is_stable else "Unstable"
                
                # Create progress bar
                progress = int(gpu_util / 5)  # 20 segments for 100%
                bar = "=" * progress + "-" * (20 - progress)
                
                # Display current status with all metrics
                print(f"\rGPU {gpu_index} [{bar}] {gpu_util:3.1f}% ({status}) "
                      f"Target: {target_percent}% | Temp: {gpu_temp}°C | "
                      f"Freq: {gpu_freq}MHz | Power: {gpu_power:.1f}W | "
                      f"Time remaining: {remaining}s | "
                      f"{battery_stat}", end="")

                # Check if test duration has elapsed (only after stabilization)
                if start_time is not None and (current_time - start_time) >= duration:
                    stop_flag[0] = True
                
            except pynvml.NVMLError as e:
                print(f"\nError reading GPU {gpu_index} metrics: {e}")
                continue
                
        except Exception as e:
            print(f"\nError during test: {e}")
            continue

    # Print final newline after the loop ends
    print()

    # Print summary
    print("\n\nTest Summary:")
    if len(metrics['utilizations']) > 0:
        print(f"Average GPU Utilization: {np.mean(metrics['utilizations']):.1f}%")
        print(f"Average GPU Temperature: {np.mean(metrics['temperatures']):.1f}°C")
        print(f"Average GPU Frequency: {np.mean(metrics['frequencies']):.1f}MHz")

        try:
            max_graphics_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            print(f"Maximum GPU Frequency: {max_graphics_clock}MHz")
            print(f"Average/Max GPU Frequency Ratio: {(np.mean(metrics['frequencies'])/max_graphics_clock):.1%}")
        except pynvml.NVMLError as e:
            print(f"Unable to get maximum GPU frequency: {e}")
        
        if len(metrics['power_values']) > 0:
            print(f"Average GPU Power: {np.mean(metrics['power_values']):.1f}W")
            try:
                max_power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
                print(f"GPU Power Limit: {max_power_limit:.1f}W")
                print(f"Average/Max Power Ratio: {(np.mean(metrics['power_values'])/max_power_limit):.1%}")
            except pynvml.NVMLError as e:
                print(f"Unable to get GPU power limit: {e}")

        # Calculate throttling score
        try:
            gamma = 1 - np.array(metrics['frequencies']) / max_graphics_clock * (1 - np.array(metrics['utilizations']) / 100)
            score_gamma = np.mean(np.exp(- gamma))
            print(f"Score_gamma: {score_gamma:.5f}, gamma: {np.mean(gamma):.5f}")
            print(f"If Score_gamma approaches 1 (gamma approaches 0), GPU {gpu_index} throttling is insignificant")
            print(f"If Score_gamma approaches 0 (gamma approaches 1), GPU {gpu_index} throttling is significant")
        except Exception as e:
            print(f"Unable to calculate throttling score: {e}")
    else:
        print("No data was collected during the test.")
            
    return metrics

class SimpleGPUStressTest:
    """Simple GPU stress test using basic arithmetic operations with dynamic load adjustment"""
    def __init__(self, device_id: int = 0):
        """
        Initialize GPU stress test
        
        Args:
            device_id (int): GPU device ID
        """
        self.device_id = device_id
        self.stream = cp.cuda.Stream()
        cp.cuda.Device(device_id).use()
        
        # Initialize workload parameters
        self.current_elements = 10**7  # Initial workload size
        self.min_elements = 10**6      # Minimum workload size
        self.max_elements = 10**9      # Maximum workload size
        
        # Load adjustment parameters
        self.adjustment_factor = 0.1    # Base adjustment factor
        self.fine_adjustment = 0.05     # Fine adjustment factor for small corrections
        self.coarse_adjustment = 0.2    # Coarse adjustment factor for large corrections
        
    def perform_computation(self, current_util: float, target_util: float) -> float:
        """
        Perform GPU computation with dynamic workload adjustment
        
        Args:
            current_util (float): Current GPU utilization percentage
            target_util (float): Target GPU utilization percentage
        
        Returns:
            float: Result of computation
        """
        # Calculate utilization difference
        util_diff = target_util - current_util
        
        # Determine adjustment strategy based on difference magnitude
        if abs(util_diff) > 2:  # Only adjust if difference is significant
            # Use different adjustment factors based on the difference
            if abs(util_diff) > 10:
                # Large difference - use coarse adjustment
                adjustment = 1 + (util_diff / 100.0 * self.coarse_adjustment)
            else:
                # Small difference - use fine adjustment
                adjustment = 1 + (util_diff / 100.0 * self.fine_adjustment)
            
            # Apply adjustment with bounds checking
            new_elements = int(self.current_elements * adjustment)
            self.current_elements = max(self.min_elements, min(self.max_elements, new_elements))
        
        return self.perform_computation_fixed()
        
    def perform_computation_fixed(self) -> float:
        """
        Perform GPU computation with current workload size
        
        Returns:
            float: Result of computation
        """
        with self.stream:
            try:
                # Create array on GPU
                x = cp.arange(self.current_elements, dtype=cp.float32)
                # Compute square sum
                x = cp.sum(x * x)
                # Ensure computation is complete
                self.stream.synchronize()
                return float(x)
            except Exception as e:
                print(f"Error in GPU computation: {e}")
                # Reduce workload size in case of errors
                self.current_elements = max(self.min_elements, self.current_elements // 2)
                return 0.0


class RayTracingStressTest:
    """Ray tracing based GPU stress test using CUDA"""
    def __init__(self, device_id: int = 0):
        """Initialize ray tracing stress test"""
        self.device_id = device_id
        self.stream = cp.cuda.Stream()
        cp.cuda.Device(device_id).use()
        
        # Start with very small initial values
        self.width = 64        # Smaller initial resolution
        self.height = 64
        self.num_spheres = 2    # Fewer initial spheres
        self.max_depth = 3
        
        # Add safety limits
        self.min_resolution = 16
        self.max_resolution = 2048  # Lower maximum resolution
        self.min_spheres = 1
        self.max_spheres = 20    # Fewer maximum spheres
        
        # Add workload adjustment parameters
        self.adjustment_step = 8     # Smaller resolution adjustment step
        self.last_adjustment = 0     # Track last adjustment time
        self.adjustment_cooldown = 0.001  # Longer cooldown between adjustments
        self.stability_threshold = 3  # 5% threshold for stability
        self.stable_count = 0
        self.unstable_count = 0
        
        print(f"Initializing ray tracing scene with {self.num_spheres} spheres...")
        
        # Create random spheres with more controlled positions
        self.spheres_pos = cp.random.uniform(-5, 5, (self.num_spheres, 3)).astype(cp.float32)
        self.spheres_rad = cp.random.uniform(0.5, 1.5, (self.num_spheres,)).astype(cp.float32)
        self.spheres_col = cp.random.uniform(0.3, 0.7, (self.num_spheres, 3)).astype(cp.float32)

    def intersect_sphere(self, origins, directions):
        """Calculate ray-sphere intersections"""
        hits = cp.zeros((origins.shape[0], self.num_spheres, 2), dtype=cp.float32)
        
        for i in range(self.num_spheres):
            oc = origins - self.spheres_pos[i]
            a = cp.sum(directions * directions, axis=1)
            b = 2.0 * cp.sum(oc * directions, axis=1)
            c = cp.sum(oc * oc, axis=1) - self.spheres_rad[i] ** 2
            
            disc = b * b - 4 * a * c
            mask = disc >= 0
            
            sqrt_disc = cp.sqrt(cp.maximum(disc, 0))
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            
            hits[:, i, 0] = cp.where(mask, t1, cp.inf)
            hits[:, i, 1] = cp.where(mask, t2, cp.inf)
        
        return hits

    def trace_rays(self, origins, directions, depth=0):
        """Recursive ray tracing"""
        if depth >= self.max_depth:
            return cp.zeros((origins.shape[0], 3), dtype=cp.float32)
        
        hits = self.intersect_sphere(origins, directions)
        t = cp.min(hits.reshape(hits.shape[0], -1), axis=1)
        sphere_idx = cp.argmin(hits.reshape(hits.shape[0], -1), axis=1) // 2
        
        mask = t < cp.inf
        if not cp.any(mask):
            return cp.zeros((origins.shape[0], 3), dtype=cp.float32)
        
        hit_points = origins + directions * t[:, None]
        normals = hit_points - self.spheres_pos[sphere_idx]
        normals = normals / cp.linalg.norm(normals, axis=1, keepdims=True)
        
        # Simple diffuse reflection
        colors = cp.where(mask[:, None], self.spheres_col[sphere_idx] * 0.8, cp.zeros_like(normals))
        return colors

    def compute_frame(self):
        """Compute one frame of ray tracing"""
        try:
            with self.stream:
                # Generate rays
                x = cp.linspace(-1, 1, self.width, dtype=cp.float32)
                y = cp.linspace(1, -1, self.height, dtype=cp.float32)
                
                # Create coordinate grid
                X, Y = cp.meshgrid(x, y)
                
                # Calculate ray directions
                directions = cp.stack([X, Y, -cp.ones_like(X)], axis=2)
                
                # Normalize directions
                norms = cp.linalg.norm(directions, axis=2, keepdims=True)
                norms = cp.where(norms == 0, 1e-10, norms)  # Avoid division by zero
                directions /= norms
                
                # Reshape for ray tracing
                directions = directions.reshape(-1, 3)
                origins = cp.zeros_like(directions)
                
                # Perform ray tracing
                colors = self.trace_rays(origins, directions)
                colors = colors.reshape(self.height, self.width, 3)
                
                # Force synchronization
                self.stream.synchronize()
                
        except cp.cuda.memory.OutOfMemoryError:
            print(f"\nOut of memory error - reducing resolution to {self.min_resolution}x{self.min_resolution}")
            self.width = self.min_resolution
            self.height = self.min_resolution
            
        except Exception as e:
            print(f"\nError in ray tracing computation: {e}")
            self.width = self.min_resolution
            self.height = self.min_resolution

    def perform_computation(self, current_util: float, target_util: float) -> None:
        """
        Perform ray tracing computation with enhanced adaptive load control
        
        Args:
            current_util (float): Current GPU utilization percentage
            target_util (float): Target GPU utilization percentage
        """
        try:
            current_time = time.time()
            load_diff = current_util - target_util
            load_diff_percent = abs(load_diff) / target_util  # Relative difference
            
            # If load is zero, aggressively increase resolution
            if current_util == 0:
                new_width = min(self.width + self.adjustment_step * 8, self.max_resolution)
                new_height = min(self.height + self.adjustment_step * 8, self.max_resolution)
                self.width = new_width
                self.height = new_height
                self.compute_frame()
                return
                
            # Check if current load is within stability threshold
            if abs(load_diff) <= self.stability_threshold:
                self.stable_count += 1
                self.unstable_count = 0
                
                # If stable for long enough, maintain current parameters
                if self.stable_count >= 10:
                    self.compute_frame()
                    return
            else:
                self.stable_count = 0
                self.unstable_count += 1
            
            # Only adjust if system is not stable and cooldown period has passed
            if current_time - self.last_adjustment >= self.adjustment_cooldown:
                # Calculate adaptive adjustment based on load difference
                base_adjustment = self.adjustment_step
                
                # Enhanced progressive adjustment factors
                if load_diff_percent > 1.0:      # >100% off target
                    factor = 16.0
                elif load_diff_percent > 0.5:    # >50% off target
                    factor = 8.0
                elif load_diff_percent > 0.3:    # >30% off target
                    factor = 4.0
                elif load_diff_percent > 0.2:    # >20% off target
                    factor = 2.0
                elif load_diff_percent > 0.1:    # >10% off target
                    factor = 1.0
                else:                            # Fine-tuning
                    factor = 0.5
                
                # Calculate adjustment size
                adjustment = int(base_adjustment * factor)
                
                # Additional scaling based on current resolution
                resolution_factor = (self.width * self.height) / (1000 * 1000)  # Normalized to 1M pixels
                adjustment = int(adjustment * max(0.5, min(2.0, resolution_factor)))
                
                # Ensure minimum adjustment
                adjustment = max(1, adjustment)
                
                if current_util > target_util:
                    # Decrease resolution when load is too high
                    # More aggressive reduction for larger deviations
                    new_width = max(self.width - adjustment, self.min_resolution)
                    new_height = max(self.height - adjustment, self.min_resolution)
                    
                    # Add small delay for immediate load reduction if deviation is large
                    if load_diff_percent > 0.5:  # >50% over target
                        time.sleep(0.01 * load_diff_percent)
                else:
                    # Increase resolution when load is too low
                    # More conservative increase to prevent overshooting
                    increase = int(adjustment * 0.75)  # More conservative increase
                    new_width = min(self.width + increase, self.max_resolution)
                    new_height = min(self.height + increase, self.max_resolution)
                
                # Only update if parameters actually changed
                if new_width != self.width or new_height != self.height:
                    # Store last resolution before change
                    self.last_stable_resolution = (self.width, self.height)
                    
                    # Apply new resolution
                    self.width = new_width
                    self.height = new_height
                    self.last_adjustment = current_time
            
            # Perform the actual computation
            self.compute_frame()
            
        except Exception as e:
            print(f"\nError in perform_computation: {e}")
            # Revert to last stable resolution if available
            if self.last_stable_resolution:
                self.width, self.height = self.last_stable_resolution
            self.compute_frame()

    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            self.stream.synchronize()
            self.stream = None
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            print(f"Error during cleanup: {e}")

def ray_tracing_stress(stop_flag: list, target_percent: float, gpu_index: int, duration: int) -> dict:
    """
    Ray tracing based GPU stress test with warm-up and test phases
    
    Args:
        stop_flag (list): Flag to control the task execution
        target_percent (float): Target GPU usage percentage (1-100)
        gpu_index (int): Index of the GPU to stress test
        duration (int): Test duration in seconds (after warm-up)
        
    Returns:
        dict: Dictionary containing test metrics
    """
    print(f"\nInitializing ray tracing stress test on GPU {gpu_index}")
    print("This test simulates a realistic GPU workload using ray tracing computations")
    
    # Initialize metrics storage
    metrics = {
        'timestamps': [],
        'utilizations': [],
        'memory_usage': [], 
        'temperatures': [],
        'frequencies': [],
        'power_values': [],
        'battery_stats': []
    }

    try:
        # Initialize NVML and get device handle
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    except pynvml.NVMLError as e:
        raise RuntimeError(f"Failed to initialize NVML: {str(e)}")

    # Initialize GPU test object
    gpu_test = RayTracingStressTest(gpu_index)
    
    # Initialize current GPU utilization
    try:
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    except pynvml.NVMLError:
        gpu_util = 0
    
    # Warm-up phase variables
    warm_up_stable_count = 0
    WARM_UP_STABILITY_THRESHOLD = 3  # 3% threshold during warm-up
    WARM_UP_STABLE_REQUIRED = 20        # Need 20 stable readings
    is_warmed_up = False
    start_time = None
    last_status_time = time.time()
    STATUS_INTERVAL = 1.0
    
    print(f"Starting warm-up phase, targeting {target_percent}% GPU utilization...")
    
    try:
        while not stop_flag[0]:
            current_time = time.time()
            
            # Perform GPU computation based on current phase
            if not is_warmed_up:
                # Warm-up phase: Aggressive load control
                gpu_test.perform_computation(gpu_util, target_percent)
                
                # Check stability during warm-up
                if abs(gpu_util - target_percent) <= WARM_UP_STABILITY_THRESHOLD:
                    warm_up_stable_count += 1
                else:
                    warm_up_stable_count = 0
                
                # Update warm-up status periodically
                if current_time - last_status_time >= STATUS_INTERVAL:
                    print(f"\rWarm-up: Current: {gpu_util:3.1f}%, Target: {target_percent}%, "
                          f"Stability: {warm_up_stable_count}/{WARM_UP_STABLE_REQUIRED}", end="")
                    last_status_time = current_time
                
                # Check if warm-up is complete
                if warm_up_stable_count >= WARM_UP_STABLE_REQUIRED:
                    is_warmed_up = True
                    start_time = time.time()
                    print(f"\nGPU {gpu_index} has stabilized. Starting {duration} second test...")
                    metrics = {key: [] for key in metrics}
            else:
                # Test phase: Conservative load control
                load_diff = abs(gpu_util - target_percent)
                if load_diff > WARM_UP_STABILITY_THRESHOLD * 2:  # Only adjust if significantly off target
                    gpu_test.perform_computation(gpu_util, target_percent)
                else:
                    gpu_test.compute_frame()
            
            try:
                # Get GPU metrics
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_usage = (memory_info.used / memory_info.total) * 100
                gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_freq = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                
                try:
                    gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except pynvml.NVMLError:
                    gpu_power = 0
                
                # Get battery status
                battery = psutil.sensors_battery()
                battery_stat = ("Charging" if battery.power_plugged else f"Battery: {battery.percent}%") if battery else "No battery"

                # Calculate remaining time and update metrics if in test phase
                if start_time is not None:
                    elapsed = int(current_time - start_time)
                    remaining = max(0, duration - elapsed)
                    
                    # Update metrics storage
                    metrics['timestamps'].append(current_time)
                    metrics['utilizations'].append(gpu_util)
                    metrics['memory_usage'].append(memory_usage)
                    metrics['temperatures'].append(gpu_temp)
                    metrics['frequencies'].append(gpu_freq)
                    metrics['power_values'].append(gpu_power)
                    metrics['battery_stats'].append(battery_stat)
                else:
                    remaining = duration

                # Determine GPU stability status
                status = "Successfully maintain" if abs(gpu_util - target_percent) <= WARM_UP_STABILITY_THRESHOLD else "Unstable"
                
                # Create progress bar
                progress = int(gpu_util / 5)  # 20 segments for 100%
                bar = "=" * progress + "-" * (20 - progress)
                
                # Display current status with all metrics
                if is_warmed_up:
                    print(f"\rGPU {gpu_index} [{bar}] {gpu_util:3.1f}% ({status}) "
                          f"Target: {target_percent}% | Temp: {gpu_temp}°C | "
                          f"Freq: {gpu_freq}MHz | Power: {gpu_power:.1f}W | "
                          f"Time remaining: {remaining}s | "
                          f"{battery_stat}", end="")

                # Check if test duration has elapsed
                if start_time is not None and (current_time - start_time) >= duration:
                    stop_flag[0] = True

            except pynvml.NVMLError as e:
                print(f"\nError reading GPU {gpu_index} metrics: {e}")
                continue

    except KeyboardInterrupt:
        print(f"\nTest interrupted for GPU {gpu_index}")
    except Exception as e:
        print(f"\nError during test: {e}")
    finally:
        try:
            gpu_test.cleanup()
        except:
            pass

    print()  # Final newline

    # Print summary
    print("\n\nTest Summary:")
    if len(metrics['utilizations']) > 0:
        print(f"Average GPU Utilization: {np.mean(metrics['utilizations']):.1f}%")
        print(f"Average GPU Temperature: {np.mean(metrics['temperatures']):.1f}°C")
        print(f"Average GPU Frequency: {np.mean(metrics['frequencies']):.1f}MHz")

        try:
            max_graphics_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            print(f"Maximum GPU Frequency: {max_graphics_clock}MHz")
            print(f"Average/Max GPU Frequency Ratio: {(np.mean(metrics['frequencies'])/max_graphics_clock):.1%}")
        except pynvml.NVMLError as e:
            print(f"Unable to get maximum GPU frequency: {e}")
        
        if len(metrics['power_values']) > 0:
            print(f"Average GPU Power: {np.mean(metrics['power_values']):.1f}W")
            try:
                max_power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
                print(f"GPU Power Limit: {max_power_limit:.1f}W")
                print(f"Average/Max Power Ratio: {(np.mean(metrics['power_values'])/max_power_limit):.1%}")
            except pynvml.NVMLError as e:
                print(f"Unable to get GPU power limit: {e}")

        # Calculate throttling score
        try:
            gamma = 1 - np.array(metrics['frequencies']) / max_graphics_clock * (1 - np.array(metrics['utilizations']) / 100)
            score_gamma = np.mean(np.exp(- gamma))
            print(f"Score_gamma: {score_gamma:.5f}, gamma: {np.mean(gamma):.5f}")
            print(f"If Score_gamma approaches 1 (gamma approaches 0), GPU {gpu_index} throttling is insignificant")
            print(f"If Score_gamma approaches 0 (gamma approaches 1), GPU {gpu_index} throttling is significant")
        except Exception as e:
            print(f"Unable to calculate throttling score: {e}")
    else:
        print("No data was collected during the test.")
            
    return metrics


class FrequencyMaxTest:
    """
    GPU Frequency Max Test - Attempts to maximize GPU frequency with minimal memory impact
    """
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        
        try:
            # Get maximum supported clock
            self.max_clock = pynvml.nvmlDeviceGetMaxClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
            print(f"Maximum GPU frequency: {self.max_clock}MHz")
            
            # Initialize CUDA resources with compute-focused parameters
            self.stream = cp.cuda.Stream(non_blocking=True)
            
            # Optimized parameters for maximum frequency
            self.num_blocks = 128        # Moderate block count
            self.threads_per_block = 256 # Moderate thread count
            self.compute_intensity = 5000 # Very high compute intensity
            self.memory_stride = 64      # Minimal memory pressure
            self.sync_frequency = 1000   # Minimal synchronization
            
            # Minimize memory footprint
            self.data_size = self.num_blocks * self.threads_per_block
            self.input_data = cp.random.rand(self.data_size, dtype=cp.float32)
            self.output_data = cp.zeros(self.data_size, dtype=cp.float32)
            self.shared_data = cp.zeros(self.threads_per_block, dtype=cp.float32)
            
            # Basic state tracking
            self.current_iterations = 0
            self.stop_background = False
            self.background_thread = None
            
            # Create CUDA kernel focused on compute intensity
            self.kernel = cp.RawKernel(r'''
            extern "C" __global__
            void frequency_max(float* input, float* output, float* shared, 
                             int n, int intensity, int stride, int sync_freq) {
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                int local_tid = threadIdx.x;
                
                __shared__ float s_data[1024];
                
                if (tid < n) {
                    float x = input[tid];
                    s_data[local_tid] = x;
                    
                    __syncthreads();
                    
                    // Compute-intensive loop with minimal memory access
                    for(int i = 0; i < intensity; i++) {
                        // Heavy compute operations
                        x = __sinf(x) * __cosf(x);
                        x = __expf(x) * __logf(fabsf(x) + 1.0f);
                        x = __powf(x, 2.0f);
                        x = __fdividef(1.0f, x + 1.0f);
                        x = __tanf(x);
                        x = __powf(x, 3.0f);
                        
                        // Minimal memory operations
                        if (i % stride == 0) {
                            int idx = (local_tid + i) % blockDim.x;
                            x += s_data[idx];
                        }
                        
                        // Reduced synchronization
                        if (i % sync_freq == 0) {
                            output[tid] = x;
                        }
                    }
                    
                    output[tid] = x;
                    shared[local_tid] = s_data[local_tid];
                }
            }
            ''', 'frequency_max')
            
        except Exception as e:
            print(f"Initialization error: {e}")
            raise

    def _background_compute(self):
        """Background computation task"""
        while not self.stop_background:
            try:
                with self.stream:
                    self.kernel(
                        (self.num_blocks,),
                        (self.threads_per_block,),
                        (self.input_data, self.output_data, self.shared_data,
                         self.data_size, self.compute_intensity,
                         self.memory_stride, self.sync_frequency)
                    )
                    self.stream.synchronize()
                self.current_iterations += 1
            except Exception as e:
                print(f"Background computation error: {e}")
                time.sleep(0.1)

    def start_background_task(self):
        """Start background computation"""
        self.stop_background = False
        self.background_thread = Thread(target=self._background_compute)
        self.background_thread.daemon = True
        self.background_thread.start()

    def stop_background_task(self):
        """Stop background computation"""
        if self.background_thread:
            self.stop_background = True
            self.background_thread.join(timeout=1.0)

    def perform_computation(self):
        """Execute computation"""
        try:
            with self.stream:
                self.kernel(
                    (self.num_blocks,),
                    (self.threads_per_block,),
                    (self.input_data, self.output_data, self.shared_data,
                     self.data_size, self.compute_intensity,
                     self.memory_stride, self.sync_frequency)
                )
                self.stream.synchronize()
            
            # Minimal delay
            time.sleep(0.005)
            
        except Exception as e:
            print(f"Computation error: {e}")
            raise

    def get_current_freq(self):
        """Get current GPU frequency"""
        try:
            return pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
        except Exception as e:
            print(f"Error getting frequency: {e}")
            return 0

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.stop_background_task()
            self.input_data = None
            self.output_data = None
            self.shared_data = None
            self.stream = None
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            print(f"Cleanup error: {e}")


def frequency_stress(stop_flag: list, target_percent: float, gpu_index: int, duration: int) -> dict:
    """
    Main frequency stress test function that aims to maximize GPU frequency
    
    Args:
        stop_flag (list): Control flag for test execution
        target_percent (float): Target GPU frequency percentage (ignored in max mode)
        gpu_index (int): GPU device index
        duration (int): Test duration in seconds
    """
    print(f"\nInitializing frequency max test on GPU {gpu_index}")
    
    metrics = {
        'timestamps': [],
        'utilizations': [],
        'memory_usage': [], 
        'temperatures': [],
        'frequencies': [],
        'power_values': [],
        'battery_stats': [],
        'memory_info': []
    }

    try:
        try:
            pynvml.nvmlInit()
        except:
            print("NVML already initialized")
            
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        gpu_test = FrequencyMaxTest(gpu_index)
        
        max_graphics_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        print(f"\nMaximum GPU frequency: {max_graphics_clock}MHz")
        
        start_time = time.time()
        last_metrics_time = start_time
        last_status_time = start_time
        
        gpu_test.start_background_task()
        
        while not stop_flag[0]:
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = max(0, duration - elapsed)
            
            if elapsed >= duration:
                break
                
            # Collect metrics every 100ms
            if current_time - last_metrics_time >= 0.1:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_freq = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    try:
                        gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except:
                        gpu_power = 0
                    
                    # Get battery status
                    battery = psutil.sensors_battery()
                    battery_stat = ("Charging" if battery.power_plugged else f"Battery: {battery.percent}%") if battery else "No battery"
                        
                    metrics['timestamps'].append(elapsed)
                    metrics['utilizations'].append(util.gpu)
                    metrics['memory_usage'].append(util.memory)
                    metrics['temperatures'].append(gpu_temp)
                    metrics['frequencies'].append(gpu_freq)
                    metrics['power_values'].append(gpu_power)
                    metrics['battery_stats'].append(battery_stat)
                    metrics['memory_info'].append({
                        'used': mem_info.used / 1024**2,  # Convert to MB
                        'free': mem_info.free / 1024**2,
                        'total': mem_info.total / 1024**2
                    })
                    
                except Exception as e:
                    print(f"Error collecting metrics: {e}")
                
                last_metrics_time = current_time
            
            # Update status display every second
            if current_time - last_status_time >= 1.0:
                freq_ratio = (gpu_freq / max_graphics_clock) * 100
                progress = int(freq_ratio / 5)
                bar = "=" * progress + "-" * (20 - progress)
                
                mem_used_mb = mem_info.used / 1024**2
                mem_total_mb = mem_info.total / 1024**2
                mem_percent = (mem_used_mb / mem_total_mb) * 100
                
                print(f"\rGPU {gpu_index} [{bar}] "
                      f"Current: {freq_ratio:.1f}% ({gpu_freq}MHz) | "
                      f"Max: {max_graphics_clock}MHz | "
                      f"GPU: {util.gpu}% | "
                      f"Mem: {mem_percent:.1f}% ({mem_used_mb:.0f}MB) | "
                      f"Temp: {gpu_temp}°C | "
                      f"Power: {gpu_power:.1f}W | "
                      f"Time: {remaining:.1f}s | "
                      f"{battery_stat}", end="")
                      
                last_status_time = current_time

            # Perform computation
            gpu_test.perform_computation()

    except KeyboardInterrupt:
        print(f"\nTest interrupted for GPU {gpu_index}")
    except Exception as e:
        print(f"\nError during test: {e}")
        traceback.print_exc()
    finally:
        try:
            gpu_test.cleanup()
        except:
            pass

    print()  # Final newline

    # Print summary
    print("\n\nTest Summary:")
    if len(metrics['utilizations']) > 0:
        print(f"Average GPU Utilization: {np.mean(metrics['utilizations']):.1f}%")
        print(f"Average Memory Usage: {np.mean([m['used'] for m in metrics['memory_info']]):.1f}MB "
              f"({np.mean(metrics['memory_usage']):.1f}%)")
        print(f"Memory Bandwidth: {np.mean(metrics['memory_usage']):.1f}%")
        print(f"Average GPU Temperature: {np.mean(metrics['temperatures']):.1f}°C")
        print(f"Average GPU Frequency: {np.mean(metrics['frequencies']):.1f}MHz")

        try:
            max_graphics_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            print(f"Maximum GPU Frequency: {max_graphics_clock}MHz")
            print(f"Average/Max GPU Frequency Ratio: {(np.mean(metrics['frequencies'])/max_graphics_clock):.1%}")
        except pynvml.NVMLError as e:
            print(f"Unable to get maximum GPU frequency: {e}")
        
        if len(metrics['power_values']) > 0:
            print(f"Average GPU Power: {np.mean(metrics['power_values']):.1f}W")
            try:
                max_power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
                print(f"GPU Power Limit: {max_power_limit:.1f}W")
                print(f"Average/Max Power Ratio: {(np.mean(metrics['power_values'])/max_power_limit):.1%}")
            except pynvml.NVMLError as e:
                print(f"Unable to get GPU power limit: {e}")

        # Calculate throttling score
        try:
            gamma = 1 - np.array(metrics['frequencies']) / max_graphics_clock * (1 - np.array(metrics['utilizations']) / 100)
            score_gamma = np.mean(np.exp(- gamma))
            print(f"Score_gamma: {score_gamma:.5f}, gamma: {np.mean(gamma):.5f}")
            print(f"If Score_gamma approaches 1 (gamma approaches 0), GPU {gpu_index} throttling is insignificant")
            print(f"If Score_gamma approaches 0 (gamma approaches 1), GPU {gpu_index} throttling is significant")
        except Exception as e:
            print(f"Unable to calculate throttling score: {e}")
    else:
        print("No data was collected during the test.")
            
    return metrics


def main():
    """Main function to handle command line arguments and run the stress test"""
    parser = argparse.ArgumentParser(description='Advanced GPU Stress Test Tool')
    parser.add_argument('-d', '--duration', type=int, default=60,
                      help='Test duration in seconds (default: 60)')
    parser.add_argument('-t', '--target', type=float, default=50,
                      help='Default target GPU usage percentage (default: 50)')
    parser.add_argument('-p', '--parallel', action='store_true',
                      help='Run tests in parallel mode')
    parser.add_argument('-g', '--gpus', type=str,
                      help='Comma-separated list of GPU indices to test')
    parser.add_argument('-l', '--loads', type=str,
                      help='Comma-separated list of target loads for each GPU')
    parser.add_argument('-o', '--output', type=str, default=None, const='.',
                      nargs='?',
                      help='Path to save the log file')
    # Add ray tracing mode option
    parser.add_argument('-m', '--mode', choices=['matrix', 'simple', 'ray', 'frequency-max'], default='simple',
                      help='Test mode: simple (default), matrix, ray, or frequency-max') 
    
    args = parser.parse_args()

    # Process output path
    log_file = None
    if args.output is not None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"gpu_stress_test_{timestamp}.txt"
        log_file = os.path.join(args.output if args.output != '.' else os.getcwd(), 
                               filename)
        print(f"Output will be saved to: {os.path.abspath(log_file)}")

    try:
        # Initialize NVML
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        # Process GPU indices
        if args.gpus:
            try:
                gpu_indices = [int(x) for x in args.gpus.split(',')]
            except ValueError:
                print("Error: Invalid GPU indices format")
                return
        else:
            # Use all available GPUs if none specified
            gpu_indices = list(range(device_count))

        # Process GPU loads
        if args.loads:
            try:
                gpu_loads = [float(x) for x in args.loads.split(',')]
            except ValueError:
                print("Error: Invalid GPU loads format")
                return
        else:
            # Use same target for all GPUs
            gpu_loads = [args.target] * len(gpu_indices)

        # Validate inputs
        if len(gpu_loads) != len(gpu_indices):
            raise ValueError("Number of loads must match number of GPUs")
        if any(idx >= device_count for idx in gpu_indices):
            raise ValueError("Invalid GPU index specified")
        if any(not 0 <= load <= 100 for load in gpu_loads):
            raise ValueError("Load values must be between 0 and 100")

        # Setup logging if needed
        original_stdout = sys.stdout
        logger = Logger(log_file) if log_file else None
        if logger:
            sys.stdout = logger

        try:
            # Print test configuration
            print("\nGPU Stress Test Configuration:")
            print(f"Duration: {args.duration} seconds")
            print(f"Mode: {'Parallel' if args.parallel else 'Serial'} {args.mode} test")
            for _, (gpu, load) in enumerate(zip(gpu_indices, gpu_loads)):
                print(f"GPU {gpu}: Target {load}%")
            print("\nInitializing test...\n")

            # Select test mode and run
            if args.parallel:
                threads = []
                shared_stop_flag = [False]

                # Create test function based on mode
                if args.mode == 'matrix':
                    test_func = lambda gpu_idx, load: matrix_stress(
                        duration=args.duration,
                        target_percent=load,
                        gpu_index=gpu_idx,
                        log_file=log_file
                    )
                elif args.mode == 'ray':  # Add ray tracing mode
                    test_func = lambda gpu_idx, load: ray_tracing_stress(
                        [False], load, gpu_idx, args.duration
                    )
                elif args.mode == 'simple':  # simple mode
                    test_func = lambda gpu_idx, load: simple_stress(
                        [False], load, gpu_idx, args.duration
                    )
                elif args.mode == 'frequency-max':  # Add frequency mode
                    test_func = lambda gpu_idx, load: frequency_stress(
                        [False], load, gpu_idx, args.duration
                    )

                # Start all tests
                for gpu_idx, target_load in zip(gpu_indices, gpu_loads):
                    thread = Thread(
                        target=test_func,
                        args=(gpu_idx, target_load)
                    )
                    threads.append(thread)
                    thread.start()

                # Wait for all tests to complete
                for thread in threads:
                    thread.join()

            else:
                # Run tests serially
                for gpu_idx, target_load in zip(gpu_indices, gpu_loads):
                    print(f"\nTesting GPU {gpu_idx} at {target_load}% load")
                    if args.mode == 'matrix':
                        matrix_stress(
                            duration=args.duration,
                            target_percent=target_load,
                            gpu_index=gpu_idx,
                            log_file=log_file
                        )
                    elif args.mode == 'ray':  # Add ray tracing mode
                        ray_tracing_stress(
                            [False], target_load, gpu_idx, args.duration
                        )
                    elif args.mode == 'simple':
                        simple_stress(
                            [False], target_load, gpu_idx, args.duration
                        )
                    elif args.mode == 'frequency-max':  # Add frequency mode
                        frequency_stress(
                            [False], target_load, gpu_idx, args.duration
                        )

        finally:
            if logger:
                sys.stdout = original_stdout
                logger.close()
            try:
                pynvml.nvmlShutdown()
            except:
                pass

    except Exception as e:
        print(f"Error during stress test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

    # Matrix mode examples:
    # Test all GPUs with default settings
    # python gpu_stress_test.py -d 60 -t 15

    # Test all GPUs in parallel with same load
    # python gpu_stress_test.py -d 60 -t 80 -p

    # Test specific GPUs with same load
    # python gpu_stress_test.py -d 60 -t 70 -g 0,2 -p

    # Test all GPUs with different loads
    # python gpu_stress_test.py -d 60 -l 60,70,80,90

    # Test specific GPUs with different loads in parallel
    # python gpu_stress_test.py -d 60 -l 60,70 -g 0,1 -p

    # Test with output logging
    # python gpu_stress_test.py -d 60 -t 80 -o ./logs
    # python gpu_stress_test.py -d 60 -l 60,70 -g 0,1 -p -o ./logs

    # Simple mode examples:
    # Test single GPU with simple arithmetic
    # python gpu_stress_test.py -d 60 -t 80 -m simple

    # Test all GPUs in parallel with same load
    # python gpu_stress_test.py -d 60 -t 80 -m simple -p

    # Test specific GPUs serially
    # python gpu_stress_test.py -d 60 -g 0,1 -t 80 -m simple

    # Test specific GPUs with different loads
    # python gpu_stress_test.py -d 60 -l 60,70 -g 0,1 -m simple

    # Test all GPUs in parallel with different loads
    # python gpu_stress_test.py -d 60 -l 60,70,80,90 -m simple -p

    # Test with output logging in simple mode
    # python gpu_stress_test.py -d 60 -t 80 -m simple -o ./logs
    # python gpu_stress_test.py -d 60 -l 60,70 -g 0,1 -m simple -p -o ./logs

    # Ray tracing mode examples:
    # Test single GPU with ray tracing
    # python gpu_stress_test.py -d 60 -t 80 -m ray

    # Test all GPUs in parallel with ray tracing
    # python gpu_stress_test.py -d 60 -t 80 -m ray -p

    # Test specific GPUs with ray tracing
    # python gpu_stress_test.py -d 60 -g 0,1 -t 80 -m ray

    # Test specific GPUs with different loads in ray tracing mode
    # python gpu_stress_test.py -d 60 -l 60,70 -g 0,1 -m ray -p

    # Combined parameter examples:
    # Long duration test with specific GPUs and loads
    # python gpu_stress_test.py -d 3600 -l 70,80 -g 0,1 -p -o ./logs -m simple

    # Quick test with all GPUs at low load
    # python gpu_stress_test.py -d 30 -t 20 -p -m simple

    # Extended stress test with high load
    # python gpu_stress_test.py -d 1800 -t 95 -p -o ./stress_test_logs