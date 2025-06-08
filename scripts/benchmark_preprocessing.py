#!/usr/bin/env python3
"""
Benchmark script to compare original vs optimized preprocessing performance

Usage:
    python scripts/benchmark_preprocessing.py
"""

import time
import hydra
from omegaconf import DictConfig
import os
from pathlib import Path


def run_benchmark_subset(cfg, num_files=100):
    """
    Run preprocessing on a subset of files for benchmarking
    """
    print(f"=== BENCHMARKING WITH {num_files} FILES ===")

    # Get list of obj files
    mesh_dir = Path(cfg.preprocessing.lidar.mesh_dir)
    obj_files = list(mesh_dir.glob("*.obj"))[:num_files]

    if len(obj_files) < num_files:
        print(f"Warning: Only {len(obj_files)} files available, using all")
        num_files = len(obj_files)

    # Create temporary mesh directory with subset
    temp_mesh_dir = mesh_dir.parent / "benchmark_meshes"
    temp_mesh_dir.mkdir(exist_ok=True)

    # Copy subset of files (Windows-compatible)
    import shutil

    for obj_file in obj_files:
        dest = temp_mesh_dir / obj_file.name
        if not dest.exists():
            shutil.copy2(obj_file, dest)

    # Update config to use temp directory
    cfg.preprocessing.lidar.mesh_dir = str(temp_mesh_dir)

    return num_files


def benchmark_original(cfg, num_files):
    """
    Benchmark original preprocessing pipeline
    """
    print(f"\n--- BENCHMARKING ORIGINAL PIPELINE ---")

    from src.preprocessing.mesh_to_pointcloud import process_all_meshes
    from src.preprocessing.pointcloud_to_graph import process_point_cloud_files

    # Set output dirs for original
    cfg.preprocessing.lidar.output_dir = "data/benchmark_pointclouds_original"
    cfg.preprocessing.processed_dir = "data/benchmark_processed_original"

    start_time = time.time()

    try:
        # Step 1: Mesh to point cloud
        mesh_start = time.time()
        process_all_meshes(cfg)
        mesh_time = time.time() - mesh_start

        # Step 2: Point cloud to graph
        graph_start = time.time()
        process_point_cloud_files(cfg)
        graph_time = time.time() - graph_start

        total_time = time.time() - start_time

        return {
            "total_time": total_time,
            "mesh_time": mesh_time,
            "graph_time": graph_time,
            "throughput": num_files / total_time,
            "success": True,
        }

    except Exception as e:
        print(f"Original pipeline failed: {e}")
        return {
            "total_time": time.time() - start_time,
            "mesh_time": 0,
            "graph_time": 0,
            "throughput": 0,
            "success": False,
            "error": str(e),
        }


def benchmark_optimized(cfg, num_files):
    """
    Benchmark optimized preprocessing pipeline
    """
    print(f"\n--- BENCHMARKING OPTIMIZED PIPELINE ---")

    from src.preprocessing.optimized_preprocessing import run_optimized_preprocessing

    # Set output dirs for optimized
    cfg.preprocessing.lidar.output_dir = "data/benchmark_pointclouds_optimized"
    cfg.preprocessing.processed_dir = "data/benchmark_processed_optimized"

    start_time = time.time()

    try:
        run_optimized_preprocessing(cfg)
        total_time = time.time() - start_time

        return {
            "total_time": total_time,
            "throughput": num_files / total_time,
            "success": True,
        }

    except Exception as e:
        print(f"Optimized pipeline failed: {e}")
        return {
            "total_time": time.time() - start_time,
            "throughput": 0,
            "success": False,
            "error": str(e),
        }


def print_benchmark_results(original_results, optimized_results, num_files):
    """
    Print comparison of benchmark results
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS ({num_files} files)")
    print(f"{'='*60}")

    if original_results["success"]:
        print(f"Original Pipeline:")
        print(f"  Total Time:    {original_results['total_time']:.2f}s")
        print(f"  Throughput:    {original_results['throughput']:.2f} files/s")
    else:
        print(f"Original Pipeline: FAILED")

    print()

    if optimized_results["success"]:
        print(f"Optimized Pipeline:")
        print(f"  Total Time:    {optimized_results['total_time']:.2f}s")
        print(f"  Throughput:    {optimized_results['throughput']:.2f} files/s")
    else:
        print(f"Optimized Pipeline: FAILED")

    # Calculate speedup
    if original_results["success"] and optimized_results["success"]:
        speedup = original_results["total_time"] / optimized_results["total_time"]

        print(f"\nPERFORMANCE IMPROVEMENT:")
        print(f"  Speedup:      {speedup:.2f}x")
        print(
            f"  Time Saved:   {original_results['total_time'] - optimized_results['total_time']:.2f}s"
        )

    print(f"{'='*60}")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Run preprocessing benchmark
    """
    # Set benchmark parameters
    benchmark_files = 50  # Small test for benchmarking

    print("PREPROCESSING PERFORMANCE BENCHMARK")
    print(f"Testing with {benchmark_files} files")

    # Setup benchmark environment
    num_files = run_benchmark_subset(cfg, benchmark_files)

    # Disable debug mode for fair comparison
    cfg.debug = False
    cfg.preprocessing.lidar.visualize = False

    # Run benchmarks
    original_results = benchmark_original(cfg.copy(), num_files)
    optimized_results = benchmark_optimized(cfg.copy(), num_files)

    # Print results
    print_benchmark_results(original_results, optimized_results, num_files)


if __name__ == "__main__":
    main()
