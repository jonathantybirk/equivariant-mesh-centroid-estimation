#!/usr/bin/env python3
"""
GPU Optimization Test Script

Test different configurations to find optimal GPU utilization settings.
"""

import torch
import time
import subprocess
import json
from pathlib import Path


def test_batch_size(batch_size, fast_dev_run=1):
    """Test a specific batch size and measure performance"""
    print(f"\n--- Testing batch_size={batch_size} ---")

    cmd = [
        "python",
        "train_gnn_clean.py",
        "--batch_size",
        str(batch_size),
        "--fast_dev_run",
        str(fast_dev_run),
        "--precision",
        "16-mixed",
    ]

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - start_time

        # Extract iteration time from output
        lines = result.stdout.split("\n")
        it_s = None
        for line in lines:
            if "it/s" in line and "Epoch" in line:
                # Extract it/s value
                try:
                    parts = line.split("it/s")
                    if len(parts) > 0:
                        it_s_part = parts[0].split()[-1]
                        if it_s_part.replace(".", "").isdigit():
                            it_s = float(it_s_part)
                except:
                    pass
                break

        success = result.returncode == 0
        return {
            "batch_size": batch_size,
            "success": success,
            "elapsed_time": elapsed,
            "iterations_per_second": it_s,
            "stdout": result.stdout if not success else "",
            "stderr": result.stderr if not success else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "batch_size": batch_size,
            "success": False,
            "elapsed_time": 300,
            "iterations_per_second": None,
            "error": "Timeout",
        }
    except Exception as e:
        return {
            "batch_size": batch_size,
            "success": False,
            "elapsed_time": time.time() - start_time,
            "iterations_per_second": None,
            "error": str(e),
        }


def main():
    print("GPU Optimization Test")
    print("=" * 50)

    # Test different batch sizes
    batch_sizes = [8, 16, 32, 48, 64, 80, 96, 128]
    results = []

    for batch_size in batch_sizes:
        result = test_batch_size(batch_size)
        results.append(result)

        if result["success"]:
            print(f"✓ Batch size {batch_size}: {result['elapsed_time']:.1f}s")
            if result["iterations_per_second"]:
                print(f"  Iterations/sec: {result['iterations_per_second']:.3f}")
        else:
            print(f"✗ Batch size {batch_size}: FAILED")
            if "error" in result:
                print(f"  Error: {result['error']}")
            # Stop if we hit memory limits
            if "CUDA out of memory" in result.get("stderr", ""):
                print("  Hit GPU memory limit, stopping here")
                break

    # Find optimal batch size
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        # Sort by iterations per second (higher is better)
        successful_results.sort(
            key=lambda x: x["iterations_per_second"] or 0, reverse=True
        )

        print("\n" + "=" * 50)
        print("RESULTS SUMMARY")
        print("=" * 50)

        for i, result in enumerate(successful_results):
            marker = "🏆 BEST" if i == 0 else f"#{i+1}"
            print(
                f"{marker} Batch size {result['batch_size']}: "
                f"{result['iterations_per_second']:.3f} it/s "
                f"({result['elapsed_time']:.1f}s)"
            )

        best = successful_results[0]
        print(f"\n🎯 RECOMMENDATION: Use batch_size={best['batch_size']}")

        # Save results
        with open("gpu_optimization_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("📊 Results saved to gpu_optimization_results.json")
    else:
        print("\n❌ No successful configurations found!")


if __name__ == "__main__":
    main()
