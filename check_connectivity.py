import torch
import numpy as np
import os
from pathlib import Path
from collections import defaultdict, deque


def check_connectivity_from_edges(edge_index, num_nodes):
    """Check if graph is connected using BFS from edge_index tensor"""
    if num_nodes == 0:
        return True, 1, []

    # Convert edge_index to adjacency list
    adj = defaultdict(list)
    if hasattr(edge_index, "numpy"):
        edges = edge_index.numpy()
    else:
        edges = edge_index

    for i in range(edges.shape[1]):
        u, v = edges[0, i], edges[1, i]
        adj[u].append(v)
        adj[v].append(u)  # Undirected graph

    visited = set()
    components = []

    for start_node in range(num_nodes):
        if start_node not in visited:
            # BFS to find connected component
            component = []
            queue = deque([start_node])
            visited.add(start_node)

            while queue:
                node = queue.popleft()
                component.append(node)

                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            components.append(component)

    is_connected = len(components) == 1
    return is_connected, len(components), [len(comp) for comp in components]


def analyze_graph_file(file_path):
    """Analyze a single graph file for connectivity"""
    try:
        # Load the graph data
        data = torch.load(file_path, map_location="cpu")

        # Extract key information
        if isinstance(data, dict):
            # Look for common graph data keys
            edge_index = None
            num_nodes = None

            # Try to find edge_index
            for key in ["edge_index", "edges", "adj"]:
                if key in data:
                    edge_index = data[key]
                    break

            # Try to find number of nodes
            for key in ["x", "pos", "node_features", "positions"]:
                if key in data and hasattr(data[key], "shape"):
                    num_nodes = data[key].shape[0]
                    break

            # If no explicit node count, infer from edge_index
            if num_nodes is None and edge_index is not None:
                if hasattr(edge_index, "max"):
                    num_nodes = int(edge_index.max().item()) + 1
                else:
                    num_nodes = int(np.max(edge_index)) + 1

        else:
            # If data is not a dict, it might be a direct tensor
            print(f"Warning: {file_path.name} - unexpected data format: {type(data)}")
            return None

        if edge_index is None:
            return {
                "file": file_path.name,
                "error": "No edge_index found",
                "data_keys": (
                    list(data.keys()) if isinstance(data, dict) else "not_dict"
                ),
            }

        if num_nodes is None:
            return {
                "file": file_path.name,
                "error": "Could not determine number of nodes",
                "data_keys": (
                    list(data.keys()) if isinstance(data, dict) else "not_dict"
                ),
            }

        # Check connectivity
        is_connected, num_components, component_sizes = check_connectivity_from_edges(
            edge_index, num_nodes
        )

        # Count edges
        num_edges = (
            edge_index.shape[1] if hasattr(edge_index, "shape") else len(edge_index[0])
        )

        return {
            "file": file_path.name,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "is_connected": is_connected,
            "num_components": num_components,
            "component_sizes": component_sizes,
            "largest_component": max(component_sizes) if component_sizes else 0,
            "smallest_component": min(component_sizes) if component_sizes else 0,
            "data_keys": list(data.keys()) if isinstance(data, dict) else "not_dict",
        }

    except Exception as e:
        return {"file": file_path.name, "error": str(e)}


def examine_data_structure(data_dir, num_samples=3):
    """Examine the structure of a few sample files"""
    data_dir = Path(data_dir)
    pt_files = list(data_dir.glob("*.pt"))

    print("=== DATA STRUCTURE ANALYSIS ===")
    for i, file_path in enumerate(pt_files[:num_samples]):
        print(f"\nFile {i+1}: {file_path.name}")
        try:
            data = torch.load(file_path, map_location="cpu")

            if isinstance(data, dict):
                print(f"  Type: dict with {len(data)} keys")
                for key, value in data.items():
                    if hasattr(value, "shape"):
                        print(f"    {key}: shape {value.shape}, dtype {value.dtype}")
                    elif hasattr(value, "__len__"):
                        print(f"    {key}: length {len(value)}, type {type(value)}")
                    else:
                        print(f"    {key}: {type(value)} = {value}")
            else:
                print(f"  Type: {type(data)}")
                if hasattr(data, "shape"):
                    print(f"  Shape: {data.shape}")

        except Exception as e:
            print(f"  Error: {e}")
    print()


def main():
    data_dir = "data/processed_sh4"

    print("=== K-NN GRAPH CONNECTIVITY CHECKER ===\n")

    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return

    # Examine data structure first
    examine_data_structure(data_dir)

    # Get all files
    data_dir = Path(data_dir)
    pt_files = list(data_dir.glob("*.pt"))

    if not pt_files:
        print(f"No .pt files found in {data_dir}")
        return

    print(f"Found {len(pt_files)} files. Checking connectivity...\n")

    # Analyze files
    results = []
    errors = []

    # Check first 20 files for quick analysis
    sample_files = pt_files[:20] if len(pt_files) > 20 else pt_files

    for file_path in sample_files:
        result = analyze_graph_file(file_path)

        if "error" in result:
            errors.append(result)
            print(f"{result['file'][:40]:40} | ERROR: {result['error']}")
        else:
            results.append(result)
            status = (
                "✓ Connected"
                if result["is_connected"]
                else f"✗ {result['num_components']} components"
            )
            print(
                f"{result['file'][:40]:40} | {result['num_nodes']:4d} nodes | {result['num_edges']:5d} edges | {status}"
            )

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Files analyzed: {len(results)}")
    print(f"Connected graphs: {sum(1 for r in results if r['is_connected'])}")
    print(f"Disconnected graphs: {sum(1 for r in results if not r['is_connected'])}")
    print(f"Errors: {len(errors)}")

    if results:
        avg_nodes = np.mean([r["num_nodes"] for r in results])
        avg_edges = np.mean([r["num_edges"] for r in results])
        avg_components = np.mean([r["num_components"] for r in results])
        print(f"Average nodes per graph: {avg_nodes:.1f}")
        print(f"Average edges per graph: {avg_edges:.1f}")
        print(f"Average components per graph: {avg_components:.1f}")

    # Show disconnected examples
    disconnected = [r for r in results if not r["is_connected"]]
    if disconnected:
        print(f"\n=== DISCONNECTED EXAMPLES ===")
        for r in disconnected[:5]:
            print(
                f"{r['file']}: {r['num_components']} components, sizes: {r['component_sizes']}"
            )

    # Show errors
    if errors:
        print(f"\n=== ERRORS ===")
        for error in errors[:5]:
            print(f"{error['file']}: {error['error']}")
            if "data_keys" in error:
                print(f"  Available keys: {error['data_keys']}")


if __name__ == "__main__":
    main()
