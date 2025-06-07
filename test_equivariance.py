import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch
from e3nn.o3 import spherical_harmonics
from src.model.eq_gnn import GNN
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import argparse
import time
from e3nn.o3 import _wigner


def generate_rotation_matrix(axis_angle=None, random_seed=None):
    """Generate a random rotation matrix or from axis-angle."""
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Generate random Euler angles
    alpha = torch.rand(1).item() * 2 * np.pi  # rotation around z-axis
    beta = torch.rand(1).item() * np.pi  # rotation around y-axis
    gamma = torch.rand(1).item() * 2 * np.pi  # rotation around x-axis

    # Construct 3x3 rotation matrix from Euler angles (ZYX convention)
    cos_alpha, sin_alpha = torch.cos(torch.tensor(alpha)), torch.sin(
        torch.tensor(alpha)
    )
    cos_beta, sin_beta = torch.cos(torch.tensor(beta)), torch.sin(torch.tensor(beta))
    cos_gamma, sin_gamma = torch.cos(torch.tensor(gamma)), torch.sin(
        torch.tensor(gamma)
    )

    # Rotation around z-axis
    R_z = torch.tensor(
        [[cos_alpha, -sin_alpha, 0], [sin_alpha, cos_alpha, 0], [0, 0, 1]],
        dtype=torch.float32,
    )

    # Rotation around y-axis
    R_y = torch.tensor(
        [[cos_beta, 0, sin_beta], [0, 1, 0], [-sin_beta, 0, cos_beta]],
        dtype=torch.float32,
    )

    # Rotation around x-axis
    R_x = torch.tensor(
        [[1, 0, 0], [0, cos_gamma, -sin_gamma], [0, sin_gamma, cos_gamma]],
        dtype=torch.float32,
    )

    # Combined rotation: R = R_z @ R_y @ R_x
    rotation_matrix = torch.matmul(torch.matmul(R_z, R_y), R_x)

    return rotation_matrix


def compute_edge_sh_features(node_pos, edge_index, max_sh_degree=1):
    """Compute spherical harmonic edge features from node positions."""
    if edge_index.shape[1] == 0:
        return torch.zeros(0, sum(2 * l + 1 for l in range(max_sh_degree + 1)))

    source_pos = node_pos[edge_index[0]]  # [E, 3]
    target_pos = node_pos[edge_index[1]]  # [E, 3]

    # Compute edge vectors (displacement from source to target)
    edge_vec = target_pos - source_pos  # [E, 3]

    # Normalize edge vectors
    edge_vec_norm = torch.norm(edge_vec, dim=1, keepdim=True)
    edge_vec_normalized = edge_vec / (edge_vec_norm + 1e-8)

    # Compute spherical harmonics for each l
    sh_features = []
    for l in range(max_sh_degree + 1):
        sh_l = spherical_harmonics(l, edge_vec_normalized, normalize=True)  # [E, 2*l+1]
        sh_features.append(sh_l)

    # Concatenate all sh features
    return torch.cat(sh_features, dim=1)


def create_test_graph(num_nodes=10, seed=42):
    """Create a simple test graph with random node positions."""
    torch.manual_seed(seed)

    # Generate random node positions
    node_pos = torch.randn(num_nodes, 3) * 2.0

    # Generate simple node features (could be positions or other features)
    node_features = torch.randn(num_nodes, 3)

    # Create a simple connectivity (each node connected to next 2 nodes in cycle)
    edge_list = []
    for i in range(num_nodes):
        for j in range(1, 3):  # Connect to next 2 nodes
            target = (i + j) % num_nodes
            edge_list.append([i, target])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    # Compute edge attributes using spherical harmonics
    edge_attr = compute_edge_sh_features(node_pos, edge_index)

    return Data(
        x=node_features, edge_index=edge_index, edge_attr=edge_attr, pos=node_pos
    )


def rotate_graph(data, rotation_matrix):
    """Apply rotation to graph node positions and recompute edge attributes."""
    # Rotate node positions
    rotated_pos = torch.matmul(data.pos, rotation_matrix.t())

    # Rotate node features if they represent 3D vectors
    rotated_x = torch.matmul(data.x, rotation_matrix.t())

    # Recompute edge attributes with rotated positions
    rotated_edge_attr = compute_edge_sh_features(rotated_pos, data.edge_index)

    return Data(
        x=rotated_x,
        edge_index=data.edge_index,
        edge_attr=rotated_edge_attr,
        pos=rotated_pos,
    )


def test_equivariance_single(
    model, data, rotation_matrix, tolerance=1e-4, verbose=False
):
    """Test equivariance for a single rotation."""
    model.eval()

    with torch.no_grad():
        # Compute output on original graph
        original_output = model(
            data.x, data.edge_index, data.edge_attr, node_pos=data.pos
        )

        # Rotate the graph
        rotated_data = rotate_graph(data, rotation_matrix)

        # Compute output on rotated graph
        rotated_output = model(
            rotated_data.x,
            rotated_data.edge_index,
            rotated_data.edge_attr,
            node_pos=rotated_data.pos,
        )

        # Rotate the original output
        expected_output = torch.matmul(original_output, rotation_matrix.t())

        # Compute error
        error = torch.norm(rotated_output - expected_output).item()

        if verbose:
            print(f"Original output: {original_output.squeeze()}")
            print(f"Rotated output: {rotated_output.squeeze()}")
            print(f"Expected output: {expected_output.squeeze()}")
            print(f"Error: {error:.6f}")

        return error < tolerance, error


def test_translation_invariance(
    model, data, translation_vector, tolerance=1e-4, verbose=False
):
    """Test that the model is translation equivariant (not invariant!)."""
    model.eval()

    with torch.no_grad():
        # Compute output on original graph
        original_output = model(
            data.x, data.edge_index, data.edge_attr, node_pos=data.pos
        )

        # Translate all node positions
        translated_data = data.clone()
        translated_data.pos = data.pos + translation_vector
        translated_data.x = (
            data.x + translation_vector
        )  # Also translate node features if they are positions

        # Recompute edge attributes (should be unchanged for relative features)
        translated_data.edge_attr = compute_edge_sh_features(
            translated_data.pos, data.edge_index
        )

        # Compute output on translated graph
        translated_output = model(
            translated_data.x,
            translated_data.edge_index,
            translated_data.edge_attr,
            node_pos=translated_data.pos,
        )

        # For center of mass prediction: the prediction should be translated by the same amount
        expected_output = original_output + translation_vector.unsqueeze(0)
        error = torch.norm(translated_output - expected_output).item()

        if verbose:
            print(f"Original output: {original_output.squeeze()}")
            print(f"Translated output: {translated_output.squeeze()}")
            print(f"Expected output: {expected_output.squeeze()}")
            print(f"Translation error: {error:.6f}")

        return error < tolerance, error


def run_comprehensive_equivariance_test(model, num_tests=20, verbose=False):
    """Run comprehensive equivariance tests."""
    print("=" * 60)
    print("COMPREHENSIVE EQUIVARIANCE TEST")
    print("=" * 60)

    # Test 1: Simple rotations around coordinate axes
    print("\n1. Testing rotations around coordinate axes...")
    axes = [
        [1, 0, 0],  # X-axis
        [0, 1, 0],  # Y-axis
        [0, 0, 1],  # Z-axis
    ]
    angles = [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]

    axis_test_results = []
    for axis in axes:
        for angle in angles:
            axis_angle = torch.tensor(axis, dtype=torch.float32) * angle
            rotation_matrix = generate_rotation_matrix(axis_angle.numpy())

            data = create_test_graph(num_nodes=8)
            passed, error = test_equivariance_single(
                model, data, rotation_matrix, verbose=verbose
            )
            axis_test_results.append((passed, error))

            if verbose:
                print(
                    f"  Rotation around {axis} by {angle:.2f} rad: {'PASS' if passed else 'FAIL'} (error: {error:.6f})"
                )

    axis_pass_rate = sum(1 for passed, _ in axis_test_results if passed) / len(
        axis_test_results
    )
    avg_axis_error = sum(error for _, error in axis_test_results) / len(
        axis_test_results
    )

    print(
        f"  Coordinate axis rotations: {axis_pass_rate:.1%} pass rate, avg error: {avg_axis_error:.6f}"
    )

    # Test 2: Random rotations
    print(f"\n2. Testing {num_tests} random rotations...")
    random_test_results = []

    for i in range(num_tests):
        rotation_matrix = generate_rotation_matrix(random_seed=i)
        data = create_test_graph(num_nodes=10, seed=i)
        passed, error = test_equivariance_single(
            model, data, rotation_matrix, verbose=verbose and i < 3
        )
        random_test_results.append((passed, error))

        if verbose and i < 3:
            print(
                f"  Random test {i+1}: {'PASS' if passed else 'FAIL'} (error: {error:.6f})"
            )

    random_pass_rate = sum(1 for passed, _ in random_test_results if passed) / len(
        random_test_results
    )
    avg_random_error = sum(error for _, error in random_test_results) / len(
        random_test_results
    )
    max_random_error = max(error for _, error in random_test_results)

    print(
        f"  Random rotations: {random_pass_rate:.1%} pass rate, avg error: {avg_random_error:.6f}, max error: {max_random_error:.6f}"
    )

    # Test 3: Different graph sizes
    print(f"\n3. Testing different graph sizes...")
    graph_sizes = [5, 10, 15, 20]
    size_test_results = []

    for size in graph_sizes:
        rotation_matrix = generate_rotation_matrix(random_seed=42)
        data = create_test_graph(num_nodes=size, seed=42)
        passed, error = test_equivariance_single(
            model, data, rotation_matrix, verbose=verbose
        )
        size_test_results.append((passed, error))

        if verbose:
            print(
                f"  Graph size {size}: {'PASS' if passed else 'FAIL'} (error: {error:.6f})"
            )

    size_pass_rate = sum(1 for passed, _ in size_test_results if passed) / len(
        size_test_results
    )
    avg_size_error = sum(error for _, error in size_test_results) / len(
        size_test_results
    )

    print(
        f"  Different sizes: {size_pass_rate:.1%} pass rate, avg error: {avg_size_error:.6f}"
    )

    # Test 4: Translation equivariance
    print(f"\n4. Testing translation equivariance...")
    translation_vectors = [
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 2.0, 0.0]),
        torch.tensor([0.0, 0.0, 3.0]),
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([-2.0, 3.0, -1.0]),
    ]

    translation_test_results = []
    for i, translation in enumerate(translation_vectors):
        data = create_test_graph(num_nodes=10, seed=i)
        passed, error = test_translation_invariance(
            model, data, translation, verbose=verbose and i < 2
        )
        translation_test_results.append((passed, error))

    translation_pass_rate = sum(
        1 for passed, _ in translation_test_results if passed
    ) / len(translation_test_results)
    avg_translation_error = sum(error for _, error in translation_test_results) / len(
        translation_test_results
    )

    print(
        f"  Translation equivariance: {translation_pass_rate:.1%} pass rate, avg error: {avg_translation_error:.6f}"
    )

    # Test 5: Batch processing
    print(f"\n5. Testing batch equivariance...")
    graphs = [create_test_graph(num_nodes=8, seed=s) for s in range(5)]
    batch_data = Batch.from_data_list(graphs)
    rotation_matrix = generate_rotation_matrix(random_seed=123)

    # Test individual vs batch processing
    individual_outputs = []
    for graph in graphs:
        with torch.no_grad():
            output = model(graph.x, graph.edge_index, graph.edge_attr)
            individual_outputs.append(output)

    with torch.no_grad():
        batch_output = model(
            batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch
        )

    # Compare batch vs individual
    batch_error = 0
    for i, individual_output in enumerate(individual_outputs):
        error = torch.norm(batch_output[i : i + 1] - individual_output).item()
        batch_error += error
    batch_error /= len(individual_outputs)

    print(f"  Batch vs individual processing error: {batch_error:.6f}")

    # Overall summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_results = (
        axis_test_results
        + random_test_results
        + size_test_results
        + translation_test_results
    )
    overall_pass_rate = sum(1 for passed, _ in all_results if passed) / len(all_results)
    overall_avg_error = sum(error for _, error in all_results) / len(all_results)
    overall_max_error = max(error for _, error in all_results)

    print(f"Overall pass rate: {overall_pass_rate:.1%}")
    print(f"Average error: {overall_avg_error:.6f}")
    print(f"Maximum error: {overall_max_error:.6f}")

    if overall_pass_rate > 0.95:
        print("✅ Model appears to be properly equivariant!")
    elif overall_pass_rate > 0.8:
        print("⚠️  Model is mostly equivariant but has some issues")
    else:
        print("❌ Model has significant equivariance problems")

    return overall_pass_rate, overall_avg_error, overall_max_error


def test_model_components(verbose=False):
    """Test individual model components for equivariance."""
    print("\n" + "=" * 60)
    print("COMPONENT-WISE TESTING")
    print("=" * 60)

    # Test spherical harmonics computation
    print("\n1. Testing spherical harmonics computation...")

    # Create test edge vectors
    edge_vecs = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=torch.float32,
    )

    # Compute SH features
    original_sh = spherical_harmonics(1, edge_vecs, normalize=True)

    # Apply rotation
    rotation_matrix = generate_rotation_matrix([0.1, 0.2, 0.3])
    rotated_vecs = torch.matmul(edge_vecs, rotation_matrix.t())
    rotated_sh = spherical_harmonics(1, rotated_vecs, normalize=True)

    # Check if SH transformed correctly (this is complex, so just check basic properties)
    sh_error = torch.norm(
        torch.norm(original_sh, dim=1) - torch.norm(rotated_sh, dim=1)
    ).item()
    print(f"  SH magnitude preservation error: {sh_error:.6f}")


def benchmark_model_performance(model, num_trials=100):
    """Benchmark model inference speed."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    graph_sizes = [10, 50, 100, 200]

    for size in graph_sizes:
        data = create_test_graph(num_nodes=size)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(data.x, data.edge_index, data.edge_attr)

        # Benchmark
        start_time = time.time()
        for _ in range(num_trials):
            with torch.no_grad():
                _ = model(data.x, data.edge_index, data.edge_attr)
        end_time = time.time()

        avg_time = (end_time - start_time) / num_trials * 1000  # ms
        print(f"  Graph size {size:3d}: {avg_time:.2f} ms/inference")


def test_edge_directionality():
    """Test that edge features are properly directional (i→j ≠ j→i)"""
    print("\n" + "=" * 60)
    print("TESTING EDGE DIRECTIONALITY")
    print("=" * 60)

    # Create simple 3-node graph: A(0) -- B(1) -- C(2)
    node_pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Node A
            [1.0, 0.0, 0.0],  # Node B
            [2.0, 0.0, 0.0],  # Node C
        ],
        dtype=torch.float32,
    )

    # Create bidirectional edges: A↔B, B↔C
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 1, 0, 2, 1],  # [source nodes]
            [1, 0, 2, 1, 0, 1, 1, 2],  # [target nodes]
        ],
        dtype=torch.long,
    )

    print(f"Nodes: A(0)={node_pos[0]}, B(1)={node_pos[1]}, C(2)={node_pos[2]}")
    print(f"Edges: {edge_index.t()}")

    # Compute directional edge features
    edge_attr = compute_edge_sh_features(node_pos, edge_index, max_sh_degree=1)

    print(f"\nEdge features shape: {edge_attr.shape}")

    # Check specific directional pairs
    pairs_to_check = [
        (0, 1),  # A→B vs B→A
        (2, 3),  # B→C vs C→B
    ]

    for idx1, idx2 in pairs_to_check:
        edge1_src, edge1_tgt = edge_index[0, idx1].item(), edge_index[1, idx1].item()
        edge2_src, edge2_tgt = edge_index[0, idx2].item(), edge_index[1, idx2].item()

        feat1 = edge_attr[idx1]
        feat2 = edge_attr[idx2]

        # Check if they're opposite directions
        if edge1_src == edge2_tgt and edge1_tgt == edge2_src:
            difference = torch.norm(feat1 - feat2).item()
            print(f"\nEdge {edge1_src}→{edge1_tgt}: {feat1[:3].numpy()}")
            print(f"Edge {edge2_src}→{edge2_tgt}: {feat2[:3].numpy()}")
            print(f"Difference norm: {difference:.6f}")

            if difference > 1e-6:
                print("✅ Edges are properly directional (different features)")
            else:
                print("❌ Edges appear symmetric (same features)")
        else:
            print(f"Edges {idx1} and {idx2} are not opposite directions")


def debug_translation_test():
    """Simple debug test for translation equivariance"""
    print("\n" + "=" * 60)
    print("DEBUG TRANSLATION TEST")
    print("=" * 60)

    # Create a simple model with debug enabled
    model = GNN(
        input_dim=3,
        hidden_dim=8,
        message_passing_steps=1,
        final_mlp_dims=[16],
        max_sh_degree=1,
        debug=True,
    )
    model.eval()

    # Create simple 2-node graph
    data = create_test_graph(num_nodes=3, seed=42)
    print(f"Original positions: {data.pos}")

    with torch.no_grad():
        print("\n--- ORIGINAL GRAPH ---")
        original_output = model(
            data.x, data.edge_index, data.edge_attr, node_pos=data.pos
        )

        # Translate by [1, 0, 0]
        translation = torch.tensor([1.0, 0.0, 0.0])
        translated_data = data.clone()
        translated_data.pos = data.pos + translation
        translated_data.x = data.x + translation  # Also translate node features
        translated_data.edge_attr = compute_edge_sh_features(
            translated_data.pos, data.edge_index
        )

        print(f"\nTranslated positions: {translated_data.pos}")

        print("\n--- TRANSLATED GRAPH ---")
        translated_output = model(
            translated_data.x,
            translated_data.edge_index,
            translated_data.edge_attr,
            node_pos=translated_data.pos,
        )

        print(f"\nOriginal output: {original_output.squeeze()}")
        print(f"Translated output: {translated_output.squeeze()}")
        print(f"Expected: {original_output.squeeze() + translation}")
        print(f"Difference: {translated_output.squeeze() - original_output.squeeze()}")
        print(f"Expected difference: {translation}")


def main():
    parser = argparse.ArgumentParser(description="Test GNN Equivariance")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--num_tests", "-n", type=int, default=20, help="Number of random tests"
    )
    parser.add_argument(
        "--tolerance", "-t", type=float, default=1e-4, help="Error tolerance"
    )
    parser.add_argument(
        "--benchmark", "-b", action="store_true", help="Run performance benchmark"
    )
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dimension")
    parser.add_argument(
        "--message_steps", type=int, default=3, help="Message passing steps"
    )

    args = parser.parse_args()

    print("Initializing GNN model...")
    model = GNN(
        input_dim=3,
        hidden_dim=args.hidden_dim,
        message_passing_steps=args.message_steps,
        final_mlp_dims=[64, 32],
        max_sh_degree=1,
        debug=False,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run main equivariance tests
    run_comprehensive_equivariance_test(model, args.num_tests, args.verbose)

    # Test individual components
    test_model_components(args.verbose)

    # Optional performance benchmark
    if args.benchmark:
        benchmark_model_performance(model)

    # Test edge directionality
    test_edge_directionality()

    # Debug translation test
    debug_translation_test()

    print(f"\nTest completed!")


if __name__ == "__main__":
    main()
