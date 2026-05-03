#!/usr/bin/env python3
"""
GPU Stress Test and Benchmark Script for Jetson Orin Nano
Tests PyTorch CUDA functionality, performance, and stability
"""

import torch
import time
import sys
import argparse
from datetime import datetime

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def get_gpu_info():
    """Get and display GPU information"""
    print_header("GPU Information")
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return False
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Multiprocessors: {props.multi_processor_count}")
    
    return True

def test_basic_operations():
    """Test basic GPU operations"""
    print_header("Basic Operations Test")
    
    try:
        # Test tensor creation on GPU
        print("Creating tensors on GPU...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        print(f"✓ Tensors created: {x.shape}, {y.shape}")
        
        # Test matrix multiplication
        print("Performing matrix multiplication...")
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"✓ Matrix multiplication: {elapsed*1000:.2f} ms")
        print(f"  Result shape: {z.shape}, Result sum: {z.sum().item():.2f}")
        
        # Test element-wise operations
        print("Testing element-wise operations...")
        a = torch.randn(5000, 5000).cuda()
        start = time.time()
        b = a * 2.5 + 1.0
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"✓ Element-wise ops: {elapsed*1000:.2f} ms")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def benchmark_operations(size=2048, iterations=10):
    """Benchmark various GPU operations"""
    print_header(f"Performance Benchmark (Size: {size}x{size}, Iterations: {iterations})")
    
    results = {}
    
    # Matrix multiplication benchmark
    print("Matrix Multiplication Benchmark...")
    times = []
    for i in range(iterations):
        x = torch.randn(size, size).cuda()
        y = torch.randn(size, size).cuda()
        torch.cuda.synchronize()
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    avg_time = sum(times) / len(times)
    gflops = (2 * size**3) / (avg_time * 1e9)
    results['matmul'] = {'time': avg_time, 'gflops': gflops}
    print(f"  Average: {avg_time*1000:.2f} ms, {gflops:.2f} GFLOPS")
    
    # Convolution benchmark (if applicable)
    print("Convolution Benchmark...")
    try:
        times = []
        for i in range(iterations):
            x = torch.randn(32, 64, size//4, size//4).cuda()
            conv = torch.nn.Conv2d(64, 128, 3, padding=1).cuda()
            torch.cuda.synchronize()
            start = time.time()
            y = conv(x)
            torch.cuda.synchronize()
            times.append(time.time() - start)
        avg_time = sum(times) / len(times)
        results['conv'] = {'time': avg_time}
        print(f"  Average: {avg_time*1000:.2f} ms")
    except Exception as e:
        print(f"  Skipped: {e}")
    
    return results

def stress_test(duration=60, size=2048):
    """Run continuous stress test"""
    print_header(f"GPU Stress Test (Duration: {duration}s, Size: {size}x{size})")
    print("Press Ctrl+C to stop early")
    
    start_time = time.time()
    iteration = 0
    errors = 0
    
    try:
        while time.time() - start_time < duration:
            iteration += 1
            try:
                # Create and multiply large matrices
                x = torch.randn(size, size).cuda()
                y = torch.randn(size, size).cuda()
                z = torch.matmul(x, y)
                
                # Additional operations
                a = torch.randn(size//2, size//2).cuda()
                b = a * 2.0 + 1.0
                c = torch.sum(b)
                
                # Clear cache periodically
                if iteration % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Print progress
                elapsed = time.time() - start_time
                if iteration % 50 == 0:
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"Iteration {iteration:4d} | "
                          f"Time: {elapsed:5.1f}s | "
                          f"Memory: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
                
            except Exception as e:
                errors += 1
                print(f"❌ Error at iteration {iteration}: {e}")
                if errors > 10:
                    print("Too many errors, stopping stress test")
                    break
        
        elapsed = time.time() - start_time
        print(f"\n✓ Stress test completed!")
        print(f"  Total iterations: {iteration}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Errors: {errors}")
        print(f"  Average iteration time: {elapsed/iteration*1000:.2f} ms")
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\nStress test interrupted after {elapsed:.2f}s")
        print(f"Completed {iteration} iterations")

def memory_test():
    """Test GPU memory allocation and deallocation"""
    print_header("GPU Memory Test")
    
    try:
        # Get initial memory state
        torch.cuda.empty_cache()
        initial_allocated = torch.cuda.memory_allocated()
        initial_reserved = torch.cuda.memory_reserved()
        
        print(f"Initial - Allocated: {initial_allocated / 1024**3:.2f} GB, "
              f"Reserved: {initial_reserved / 1024**3:.2f} GB")
        
        # Allocate progressively larger tensors
        sizes = [1024, 2048, 3072, 4096]
        tensors = []
        
        for size in sizes:
            try:
                print(f"Allocating {size}x{size} tensor...")
                x = torch.randn(size, size).cuda()
                tensors.append(x)
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            except RuntimeError as e:
                print(f"  ❌ Failed: {e}")
                break
        
        # Clean up
        del tensors
        torch.cuda.empty_cache()
        final_allocated = torch.cuda.memory_allocated() / 1024**3
        final_reserved = torch.cuda.memory_reserved() / 1024**3
        
        print(f"\nAfter cleanup - Allocated: {final_allocated:.2f} GB, "
              f"Reserved: {final_reserved:.2f} GB")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='GPU Stress Test and Benchmark')
    parser.add_argument('--stress', type=int, default=0,
                       help='Run stress test for N seconds (0 = skip)')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')
    parser.add_argument('--size', type=int, default=2048,
                       help='Matrix size for tests (default: 2048)')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of benchmark iterations (default: 10)')
    parser.add_argument('--memory', action='store_true',
                       help='Run memory allocation tests')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  GPU Stress Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Check GPU availability
    if not get_gpu_info():
        sys.exit(1)
    
    # Run basic tests
    if not test_basic_operations():
        print("\n❌ Basic operations failed!")
        sys.exit(1)
    
    # Run benchmarks
    if args.benchmark:
        benchmark_operations(size=args.size, iterations=args.iterations)
    
    # Run memory tests
    if args.memory:
        memory_test()
    
    # Run stress test
    if args.stress > 0:
        stress_test(duration=args.stress, size=args.size)
    else:
        print("\n💡 Tip: Use --stress N to run a continuous stress test for N seconds")
        print("   Example: python3 test_gpu.py --stress 300")
    
    print_header("Test Complete")
    print("✓ All tests passed!")

if __name__ == '__main__':
    main()
