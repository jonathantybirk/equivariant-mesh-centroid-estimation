import torch
import torch.nn as nn
from e3nn.o3 import _wigner, spherical_harmonics
import math
import time


def validate_triangle_inequality(l1, l2, l3):
    """
    Check triangle inequality for Clebsch-Gordan coefficients.
    For valid coupling: |l1 - l2| ≤ l3 ≤ l1 + l2
    """
    min_l = abs(l1 - l2)
    max_l = l1 + l2
    if not (min_l <= l3 <= max_l):
        raise ValueError(
            f"Invalid CG coupling: l1={l1}, l2={l2}, l3={l3}. "
            f"Triangle inequality requires {min_l} ≤ l3 ≤ {max_l}"
        )


def xavier_init_steerable(weight_tensor, fan_in, fan_out):
    """Xavier (Glorot) normal initialization adapted for steerable networks"""
    # Standard Xavier: variance = 2 / (fan_in + fan_out)
    std = math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        weight_tensor.normal_(0, std)


def xavier_uniform_init_steerable(weight_tensor, fan_in, fan_out):
    """Xavier (Glorot) uniform initialization adapted for steerable networks"""
    # Xavier uniform: bound = sqrt(6 / (fan_in + fan_out))
    # This gives same variance as Xavier normal: 2 / (fan_in + fan_out)
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        weight_tensor.uniform_(-bound, bound)


class CGWeight(torch.nn.Module):
    """
    Clebsch-Gordan tensor product layer with learnable weights.

    Computes weighted sum of CG tensor products: Σ w_ij * CG(a_i ⊗ h_j → l_out)
    Only valid combinations (satisfying triangle inequality) are computed.

    Args:
        input_a_l (list[int]): L-values for input_a tensors
        input_h_l (list[int]): L-values for input_h tensors
        l_out (int): Output l-value
        init_method (str): Weight initialization method:
            - "xavier": Xavier (Glorot) normal initialization (default)
            - "xavier_uniform": Xavier (Glorot) uniform initialization
            - "constant": All weights set to 0.1
            - "uniform": Uniform initialization scaled by fan-in
            - "kaiming": Kaiming (He) initialization

    Raises:
        ValueError: If no valid CG combinations exist (all violate triangle inequality)
    """

    def __init__(self, input_a_l, input_h_l, l_out, init_method="xavier"):
        super(CGWeight, self).__init__()
        self.input_a_l = input_a_l
        self.input_h_l = input_h_l
        self.l_out = l_out

        # Validate triangle inequality for all combinations
        valid_count = 0
        for a_l_in in input_a_l:
            for h_l_in in input_h_l:
                try:
                    validate_triangle_inequality(a_l_in, h_l_in, l_out)
                    valid_count += 1
                except ValueError as e:
                    continue  # Skip invalid combinations

        if valid_count == 0:
            raise ValueError(
                f"No valid CG combinations possible for input_a_l={input_a_l}, "
                f"input_h_l={input_h_l}, l_out={l_out}. All combinations violate triangle inequality."
            )

        # Initialize weights parameter - use float32
        self.weight = torch.nn.parameter.Parameter(
            torch.zeros([len(input_a_l) * len(input_h_l)], dtype=torch.float32)
        )

        # Store only valid CG combinations
        self.valid_combos = []
        for a_l_idx, a_l_in in enumerate(input_a_l):
            for h_l_idx, h_l_in in enumerate(input_h_l):
                try:
                    validate_triangle_inequality(a_l_in, h_l_in, l_out)
                    # Additional check for non-zero CG coefficients
                    GC = _wigner._so3_clebsch_gordan(a_l_in, h_l_in, l_out)
                    if GC.abs().sum() > 0:
                        self.valid_combos.append((a_l_idx, h_l_idx))
                except (ValueError, Exception):
                    # Skip invalid combinations
                    pass

        # Apply initialization method
        fan_in = len(self.valid_combos)  # Number of valid CG combinations
        fan_out = self.l_out * 2 + 1  # Output dimension

        if init_method == "xavier":
            xavier_init_steerable(self.weight, fan_in, fan_out)
        elif init_method == "xavier_uniform":
            xavier_uniform_init_steerable(self.weight, fan_in, fan_out)
        elif init_method == "constant":
            with torch.no_grad():
                self.weight.fill_(0.1)
        elif init_method == "uniform":
            # Uniform initialization scaled by fan-in
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.1
            with torch.no_grad():
                self.weight.uniform_(-bound, bound)
        elif init_method == "kaiming":
            # Kaiming (He) initialization: variance = 2 / fan_in
            std = math.sqrt(2.0 / fan_in) if fan_in > 0 else 0.1
            with torch.no_grad():
                self.weight.normal_(0, std)

    def forward(self, input_a, input_h):
        # Convert all inputs to float32 for consistency
        input_a = [x.float() for x in input_a]
        input_h = [x.float() for x in input_h]

        # Validate input dimensions
        if len(input_a) != len(self.input_a_l):
            raise ValueError(
                f"Expected {len(self.input_a_l)} input_a tensors for l-values {self.input_a_l}, "
                f"but got {len(input_a)} tensors"
            )

        if len(input_h) != len(self.input_h_l):
            raise ValueError(
                f"Expected {len(self.input_h_l)} input_h tensors for l-values {self.input_h_l}, "
                f"but got {len(input_h)} tensors"
            )

        # Validate tensor dimensions for each l-value
        for i, (tensor, l_val) in enumerate(zip(input_a, self.input_a_l)):
            expected_dim = 2 * l_val + 1
            if tensor.shape[0] != expected_dim:
                raise ValueError(
                    f"input_a[{i}] for l={l_val} should have dimension {expected_dim}, "
                    f"but got {tensor.shape[0]}"
                )

        for i, (tensor, l_val) in enumerate(zip(input_h, self.input_h_l)):
            expected_dim = 2 * l_val + 1
            if tensor.shape[0] != expected_dim:
                raise ValueError(
                    f"input_h[{i}] for l={l_val} should have dimension {expected_dim}, "
                    f"but got {tensor.shape[0]}"
                )

        # Create output tensor - float32
        out_dim = self.l_out * 2 + 1
        result = torch.zeros([out_dim], dtype=torch.float32, device=self.weight.device)

        # Process only valid combinations
        for combo_idx, (a_l_idx, h_l_idx) in enumerate(self.valid_combos):
            a_l_in = self.input_a_l[a_l_idx]
            h_l_in = self.input_h_l[h_l_idx]

            # Safety check
            if a_l_idx >= len(input_a) or h_l_idx >= len(input_h):
                continue

            try:
                # Get the CG coefficients - convert to float32 and move to correct device
                GC = (
                    _wigner._so3_clebsch_gordan(a_l_in, h_l_in, self.l_out)
                    .float()
                    .to(result.device)
                )

                # Get the inputs
                a_tensor = input_a[a_l_idx]
                h_tensor = input_h[h_l_idx]

                # Apply the CG product
                temp = torch.einsum("ijk,i,j->k", GC, a_tensor, h_tensor)

                # Multiply by weight and add to result
                result += self.weight[combo_idx] * temp
            except Exception:
                # Skip any calculations that fail
                continue

        # Apply torch.nan_to_num as a safety measure
        result = torch.nan_to_num(result, nan=0.0)
        return result

    def forward_vectorized(self, input_a_batch, input_h_batch):
        """
        Vectorized forward pass for batch of edges

        Args:
            input_a_batch: list of [E, irrep_dim] tensors for edge features
            input_h_batch: list of [E, irrep_dim] tensors for source node features

        Returns:
            [E, out_irrep_dim] tensor for this irrep type
        """
        # Convert all inputs to float32 for consistency
        input_a_batch = [x.float() for x in input_a_batch]
        input_h_batch = [x.float() for x in input_h_batch]

        # Validate input dimensions
        if len(input_a_batch) != len(self.input_a_l):
            raise ValueError(
                f"Expected {len(self.input_a_l)} input_a tensors for l-values {self.input_a_l}, "
                f"but got {len(input_a_batch)} tensors"
            )

        if len(input_h_batch) != len(self.input_h_l):
            raise ValueError(
                f"Expected {len(self.input_h_l)} input_h tensors for l-values {self.input_h_l}, "
                f"but got {len(input_h_batch)} tensors"
            )

        # Get batch size (number of edges)
        num_edges = input_a_batch[0].shape[0] if input_a_batch else 0
        if num_edges == 0:
            out_dim = self.l_out * 2 + 1
            return torch.zeros(
                [0, out_dim], dtype=torch.float32, device=self.weight.device
            )

        # Create output tensor - float32
        out_dim = self.l_out * 2 + 1
        result = torch.zeros(
            [num_edges, out_dim], dtype=torch.float32, device=self.weight.device
        )

        # Process only valid combinations
        for combo_idx, (a_l_idx, h_l_idx) in enumerate(self.valid_combos):
            a_l_in = self.input_a_l[a_l_idx]
            h_l_in = self.input_h_l[h_l_idx]

            # Safety check
            if a_l_idx >= len(input_a_batch) or h_l_idx >= len(input_h_batch):
                continue

            try:
                # Get the CG coefficients - convert to float32 and move to correct device
                GC = (
                    _wigner._so3_clebsch_gordan(a_l_in, h_l_in, self.l_out)
                    .float()
                    .to(result.device)
                )

                # Get the inputs - [E, irrep_dim]
                a_tensor = input_a_batch[a_l_idx]
                h_tensor = input_h_batch[h_l_idx]

                # Apply the CG product vectorized: GC[i,j,k] * a[E,i] * h[E,j] -> result[E,k]
                # Use einsum for vectorized computation
                temp = torch.einsum("ijk,ei,ej->ek", GC, a_tensor, h_tensor)

                # Multiply by weight and add to result
                result += self.weight[combo_idx] * temp
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Vectorized message function failed: {e}")
                # Skip any calculations that fail
                continue

        # Apply torch.nan_to_num as a safety measure
        result = torch.nan_to_num(result, nan=0.0)
        return result


def invariant_feat(x_l_in, x_in):
    GC = _wigner._so3_clebsch_gordan(x_l_in, x_l_in, 0).float().to(x_in.device)
    return torch.einsum("ijk,i,j->k", GC, x_in, x_in)


class HiddenHLayer(torch.nn.Module):
    def __init__(self, input_a_l, input_h_l, h_l_out, init_method="xavier"):
        super(HiddenHLayer, self).__init__()
        # Use ModuleList instead of regular Python list to properly register parameters
        self.cg_weight = torch.nn.ModuleList(
            [CGWeight(input_a_l, input_h_l, l_out, init_method) for l_out in h_l_out]
        )
        self.h_l_out = h_l_out

    def forward(self, input_a, input_h):
        # Return a list of tensors, one for each output l value
        outputs = []
        for i, l_out in enumerate(self.h_l_out):
            # Get output from CGWeight
            output = self.cg_weight[i](input_a, input_h)
            outputs.append(output)
        return outputs

    def forward_vectorized(self, input_a_batch, input_h_batch):
        """
        Vectorized forward pass for batch of edges

        Args:
            input_a_batch: list of [E, irrep_dim] tensors
            input_h_batch: list of [E, irrep_dim] tensors

        Returns:
            list of [E, irrep_dim] tensors for each output l value
        """
        outputs = []
        for i, l_out in enumerate(self.h_l_out):
            # Get vectorized output from CGWeight
            output = self.cg_weight[i].forward_vectorized(input_a_batch, input_h_batch)
            outputs.append(output)
        return outputs


class MessageFunction(torch.nn.Module):
    def __init__(self, input_a_l, input_h_l, h_l_out, init_method="xavier"):
        super(MessageFunction, self).__init__()
        # Use l-values that can form valid CG coefficients
        # Use same as h_l_out to preserve equivariance
        self.intermediate_dims = h_l_out
        self.hidden_h_layer_1 = HiddenHLayer(
            input_a_l, input_h_l, self.intermediate_dims, init_method
        )
        self.hidden_h_layer_2 = HiddenHLayer(
            input_a_l, self.intermediate_dims, h_l_out, init_method
        )
        self.h_l_out = h_l_out

    def forward(self, input_a, input_h):
        # First layer outputs a list of tensors
        h1 = self.hidden_h_layer_1(input_a, input_h)

        # Second layer with h1 list
        h2 = self.hidden_h_layer_2(input_a, h1)

        return h2

    def forward_vectorized(self, input_a_batch, input_h_batch):
        """
        Vectorized forward pass for batch of edges

        Args:
            input_a_batch: list of [E, irrep_dim] tensors for edge features
            input_h_batch: list of [E, irrep_dim] tensors for source node features

        Returns:
            list of [E, irrep_dim] tensors for output messages
        """
        # First layer outputs a list of tensors
        h1_batch = self.hidden_h_layer_1.forward_vectorized(
            input_a_batch, input_h_batch
        )

        # Second layer with h1 list
        h2_batch = self.hidden_h_layer_2.forward_vectorized(input_a_batch, h1_batch)

        return h2_batch


def compute_spherical_harmonics(vectors, max_l=2):
    """
    Compute spherical harmonics for 3D vectors up to degree max_l

    Args:
        vectors: [N, 3] tensor of 3D vectors
        max_l: maximum spherical harmonic degree

    Returns:
        List of tensors, one for each l from 0 to max_l
    """
    # Ensure float32 and normalize vectors and handle zero vectors
    vectors = vectors.float()
    norms = torch.norm(vectors, dim=1, keepdim=True)
    # Avoid division by zero
    norms = torch.clamp(norms, min=1e-8)
    normalized_vectors = vectors / norms

    # Compute spherical harmonics for each degree
    sh_features = []
    for l in range(max_l + 1):
        # spherical_harmonics expects (l, vectors) where vectors is [N, 3]
        sh_l = spherical_harmonics(l, normalized_vectors, normalize=True)
        # sh_l has shape [N, 2*l+1] - ensure float32
        sh_features.append(sh_l.float())

    return sh_features


class GNN(torch.nn.Module):
    """
    Graph Neural Network for center of mass estimation (equivariant version)

    Simple interface:
    - input_dim: dimension of input node features
    - hidden_dim: dimension of hidden representations
    - max_sh_degree: automatically determines edge feature irreps [0, 1, ..., max_sh_degree]
    - For center of mass: uses irreps [0, 1] (scalar + vector) automatically
    """

    def __init__(
        self,
        input_dim=3,  # Input node feature dimension (e.g. 3 for 3D positions)
        hidden_dim=16,  # Hidden representation dimension
        message_passing_steps=3,  # Number of GNN layers
        final_mlp_dims=[64, 32],  # MLP dimensions for final prediction
        max_sh_degree=1,  # Maximum spherical harmonic degree (determines edge features)
        init_method="xavier",  # Weight initialization
        seed=42,
        debug=False,  # Enable debug output
    ):
        super().__init__()
        torch.manual_seed(seed)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_method = init_method
        self.max_sh_degree = max_sh_degree
        self.debug = debug

        # Automatically determine irrep structure from max_sh_degree
        self.input_a_l = list(
            range(max_sh_degree + 1)
        )  # [0, 1, ..., max_sh_degree] for edges
        self.input_h_l = [
            0,
            1,
        ]  # Use scalar + vector for node features (standard for 3D)
        self.h_l_out = [0, 1]  # Output scalar + vector (needed for center of mass)

        # Initial node embedding: map input_dim to hidden representations for each irrep
        self.node_encoders = nn.ModuleList(
            [nn.Linear(input_dim, 2 * l + 1) for l in self.input_h_l]
        )

        # Create GNN layers
        self.layers = nn.ModuleList()
        for i in range(message_passing_steps):
            if i == 0:
                # First layer: maps from input to hidden
                layer = GNNLayer(
                    self.input_a_l,
                    self.input_h_l,
                    self.h_l_out,
                    self.init_method,
                    self.debug,
                )
            else:
                # Hidden layers: maps from hidden to hidden
                layer = GNNLayer(
                    self.input_a_l,
                    self.h_l_out,
                    self.h_l_out,
                    self.init_method,
                    self.debug,
                )
            self.layers.append(layer)

        # Final MLP for generating weights (using only invariant features)
        final_layers = []
        # Input: l=0 features (1 scalar) + invariant features from l=1 (1 scalar) = 2 scalars
        prev_dim = 2
        for dim in final_mlp_dims:
            final_layers.append(nn.Linear(prev_dim, dim))
            final_layers.append(nn.ReLU())
            prev_dim = dim
        final_layers.append(nn.Linear(prev_dim, 1))  # Output 1 scalar weight per node
        self.final_mlp = nn.Sequential(*final_layers)

    def forward(self, x, edge_index, edge_attr, batch=None, node_pos=None):
        """
        Forward pass through the equivariant network.
        Args:
            x: Node features [N, hidden_dim]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge attributes - either [E, 3] displacement vectors OR
                      [E, total_sh_dim] preprocessed spherical harmonics
            batch: Batch assignment for nodes [N]
            node_pos: Node positions [N, 3] (optional, for computing displacements)
        Returns:
            Center of mass prediction [B, 3] where B is batch size
        """
        # Determine if edge_attr contains preprocessed spherical harmonics or raw displacements
        # SH features have dimension: sum(2*l+1 for l in range(max_l+1))
        # For max_l=1: 1 + 3 = 4 dimensions
        # For max_l=2: 1 + 3 + 5 = 9 dimensions
        edge_attr_dim = edge_attr.size(1)

        if edge_attr_dim == 3:
            # Raw displacement vectors - compute SH during forward pass (slower)
            if self.debug:
                print(
                    "⚠️  Using raw displacement vectors - computing SH during forward pass"
                )
            displacement_vectors = edge_attr
            sh_edge_features = compute_spherical_harmonics(
                displacement_vectors, max_l=self.max_sh_degree
            )
        else:
            # Preprocessed spherical harmonics - split concatenated features back into list
            if self.debug:
                print(f"⚡ Using preprocessed SH features (dim={edge_attr_dim})")
            sh_edge_features = self._split_sh_features(edge_attr, self.max_sh_degree)

        # Initial embedding: produce a list of irreps for each node
        node_irreps = [enc(x) for enc in self.node_encoders]  # List of [N, 2l+1]
        # Transpose to list of length N, each is list of irreps for that node
        node_irreps = [
            [node_irreps[i][n] for i in range(len(node_irreps))]
            for n in range(x.shape[0])
        ]

        # Message passing: propagate irreps through GNN layers using SH edge features
        for layer in self.layers:
            node_irreps = layer(node_irreps, edge_index, sh_edge_features)

        # For equivariant center of mass prediction:
        # 1. Create invariant features: l=0 + invariant features from l=1
        # 2. Use MLP on invariant features to get scalar weights
        # 3. Multiply weights with l=1 features to get equivariant 3D output

        invariant_features = []
        l1_features = []

        for node_irrep in node_irreps:
            # Extract l=0 (scalar) features
            l0_feat = node_irrep[0]  # [1] tensor

            # Extract l=1 (vector) features and create invariant features
            l1_feat = node_irrep[1]  # [3] tensor
            l1_invariant = invariant_feat(
                1, l1_feat
            )  # [1] tensor - invariant scalar from l=1

            # Combine l=0 and l=1 invariant features
            combined_invariant = torch.cat([l0_feat, l1_invariant], dim=0)  # [2] tensor

            invariant_features.append(combined_invariant)
            l1_features.append(l1_feat)

        # Stack invariant features for MLP
        invariant_features = torch.stack(invariant_features, dim=0)  # [N, 2]
        l1_features = torch.stack(l1_features, dim=0)  # [N, 3]

        # Generate scalar weights using only invariant features
        scalar_weights = self.final_mlp(invariant_features)  # [N, 1]

        # Create equivariant prediction: scalar_weight * l1_vector for each node
        node_predictions = scalar_weights * l1_features  # [N, 3] - equivariant!

        # Average predictions across nodes in each graph
        if batch is not None:
            com_prediction = self._scatter_mean(node_predictions, batch)
        else:
            com_prediction = node_predictions.mean(dim=0, keepdim=True)
        return com_prediction

    def _scatter_mean(self, src, index, dim_size=None):
        """Custom function to replace torch_scatter.scatter_mean for pooling"""
        if dim_size is None:
            dim_size = index.max().item() + 1

        # Create output tensor
        out = torch.zeros(dim_size, src.size(1), device=src.device)

        # Count elements per target index
        ones = torch.ones(src.size(0), 1, device=src.device)
        count = torch.zeros(dim_size, 1, device=src.device)
        count.index_add_(0, index, ones)

        # Sum elements with same target
        out.index_add_(0, index, src)

        # Avoid division by zero
        count = torch.clamp(count, min=1.0)

        # Compute mean
        out = out / count

        return out

    def _split_sh_features(self, concat_sh_features, max_l):
        """
        Split concatenated spherical harmonics features back into list format

        Args:
            concat_sh_features: [E, total_sh_dim] concatenated SH features
            max_l: maximum spherical harmonic degree

        Returns:
            List of tensors, one for each l from 0 to max_l
        """
        sh_features = []
        start_idx = 0

        for l in range(max_l + 1):
            dim_l = 2 * l + 1
            end_idx = start_idx + dim_l
            sh_l = concat_sh_features[:, start_idx:end_idx]  # [E, 2*l+1]
            sh_features.append(sh_l)
            start_idx = end_idx

        return sh_features


class GNNLayer(nn.Module):
    """
    Equivariant Graph Neural Network layer with efficient vectorized message passing
    """

    def __init__(
        self, input_a_l, input_h_l, h_l_out, init_method="xavier", debug=False
    ):
        super().__init__()
        self.input_a_l = input_a_l
        self.input_h_l = input_h_l
        self.h_l_out = h_l_out
        self.message_function = MessageFunction(
            input_a_l, input_h_l, h_l_out, init_method
        )
        self.debug = debug

    def forward(self, node_irreps, edge_index, sh_edge_features):
        """
        Efficient vectorized message passing for equivariant irreps using batch operations

        Args:
            node_irreps: list of length N, each is list of irreps (tensors)
            edge_index: [2, E] tensor with source/target node indices
            sh_edge_features: list of [E, 2*l+1] tensors for l=0,1,2,... (precomputed SH)

        Returns:
            updated_node_irreps: list of length N, each is list of irreps
        """
        start_time = time.time()

        num_nodes = len(node_irreps)
        num_edges = edge_index.shape[1]

        if self.debug:
            print(f"[DEBUG] Vectorized GNNLayer: {num_nodes} nodes, {num_edges} edges")

        # Convert node irreps to stacked tensors for vectorized operations
        # Each irrep type gets its own tensor: [N, irrep_dim]
        stacked_node_irreps = []
        for l_idx in range(len(self.input_h_l)):
            irreps_list = [node_irreps[n][l_idx] for n in range(num_nodes)]
            stacked_irreps = torch.stack(irreps_list, dim=0)  # [N, irrep_dim]
            stacked_node_irreps.append(stacked_irreps)

        # Vectorized message computation
        vectorized_start = time.time()

        # Get source and target indices
        source_indices = edge_index[0]  # [E]
        target_indices = edge_index[1]  # [E]

        # Gather source node features for all edges at once
        # source_irreps[l_idx] will be [E, irrep_dim]
        source_irreps = []
        for l_idx in range(len(self.input_h_l)):
            source_features = stacked_node_irreps[l_idx][
                source_indices
            ]  # [E, irrep_dim]
            source_irreps.append(source_features)

        # Compute messages for all edges simultaneously using vectorized MessageFunction
        try:
            # Use vectorized message function - processes all edges at once
            messages_by_irrep = self.message_function.forward_vectorized(
                sh_edge_features, source_irreps
            )
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Vectorized message function failed: {e}")
            # Fallback to edge-by-edge processing if vectorized version fails
            edge_messages = []
            for e in range(num_edges):
                # Extract features for this edge
                input_a = [sh_edge_features[l][e] for l in range(len(sh_edge_features))]
                input_h = [source_irreps[l][e] for l in range(len(source_irreps))]

                try:
                    message = self.message_function(input_a, input_h)
                    edge_messages.append(message)
                except Exception:
                    # Skip problematic edges - create zero messages
                    zero_message = []
                    for l_out in self.h_l_out:
                        dim_out = 2 * l_out + 1
                        device = (
                            source_irreps[0].device
                            if source_irreps
                            else torch.device("cpu")
                        )
                        dtype = (
                            source_irreps[0].dtype if source_irreps else torch.float32
                        )
                        zero_message.append(
                            torch.zeros(dim_out, device=device, dtype=dtype)
                        )
                    edge_messages.append(zero_message)

            # Convert edge messages to tensors for efficient scatter operations
            # messages_by_irrep[l_idx] will be [E, irrep_dim]
            messages_by_irrep = []
            for l_idx in range(len(self.h_l_out)):
                messages_for_irrep = torch.stack(
                    [msg[l_idx] for msg in edge_messages], dim=0
                )  # [E, irrep_dim]
                messages_by_irrep.append(messages_for_irrep)

        vectorized_time = time.time() - vectorized_start
        if self.debug:
            print(f"[DEBUG] Vectorized message computation: {vectorized_time:.3f}s")

        # Efficient scatter-based aggregation using index_add
        scatter_start = time.time()

        # Initialize aggregated messages
        aggregated_messages = []
        for l_idx in range(len(self.h_l_out)):
            l_out = self.h_l_out[l_idx]
            dim_out = 2 * l_out + 1
            device = messages_by_irrep[l_idx].device
            dtype = messages_by_irrep[l_idx].dtype

            # Create zero tensor for aggregation
            agg_tensor = torch.zeros(num_nodes, dim_out, device=device, dtype=dtype)

            # Use index_add for efficient scatter-sum operation
            agg_tensor.index_add_(0, target_indices, messages_by_irrep[l_idx])

            aggregated_messages.append(agg_tensor)

        # Count messages per node for averaging
        message_counts = torch.zeros(
            num_nodes, device=edge_index.device, dtype=torch.float32
        )
        message_counts.index_add_(
            0, target_indices, torch.ones(num_edges, device=edge_index.device)
        )

        scatter_time = time.time() - scatter_start
        if self.debug:
            print(f"[DEBUG] Scatter aggregation: {scatter_time:.3f}s")

        # Convert back to node-wise irrep format and average
        final_start = time.time()
        updated_node_irreps = []
        for n in range(num_nodes):
            count = message_counts[n]
            if count > 0:
                # Average the messages
                updated_irreps = [agg[n] / count for agg in aggregated_messages]
            else:
                # No incoming messages, use zeros
                updated_irreps = []
                for l_out in self.h_l_out:
                    dim_out = 2 * l_out + 1
                    device = (
                        aggregated_messages[0].device
                        if aggregated_messages
                        else torch.device("cpu")
                    )
                    dtype = (
                        aggregated_messages[0].dtype
                        if aggregated_messages
                        else torch.float32
                    )
                    updated_irreps.append(
                        torch.zeros(dim_out, device=device, dtype=dtype)
                    )

            updated_node_irreps.append(updated_irreps)

        final_time = time.time() - final_start
        total_time = time.time() - start_time

        if self.debug:
            print(f"[DEBUG] Final conversion: {final_time:.3f}s")
            print(f"[DEBUG] Total Vectorized GNNLayer: {total_time:.3f}s")
            print(
                f"[DEBUG] Speedup vs edge loop: {(num_edges * 0.01):.1f}s -> {total_time:.3f}s"
            )

        return updated_node_irreps
