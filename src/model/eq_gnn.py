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


class CGCoefficientsCache:
    """
    Global cache for Clebsch-Gordan coefficients to avoid recomputation
    """

    _cache = {}

    @classmethod
    def get_coefficients(cls, l1, l2, l3, device):
        """Get cached CG coefficients or compute and cache them"""
        key = (l1, l2, l3)
        if key not in cls._cache:
            try:
                cls._cache[key] = _wigner._so3_clebsch_gordan(l1, l2, l3).float()
            except Exception:
                cls._cache[key] = None

        cg_coeffs = cls._cache[key]
        if cg_coeffs is not None:
            return cg_coeffs.to(device)
        return None


class CGWeight(torch.nn.Module):
    """
    Highly optimized Clebsch-Gordan tensor product layer with precomputed coefficients.

    Key optimizations:
    1. Precompute all CG coefficients during initialization
    2. Store only valid combinations with their coefficients
    3. Use efficient batched tensor operations
    4. Minimize device transfers and memory allocations
    """

    def __init__(self, input_a_l, input_h_l, l_out, init_method="xavier"):
        super(CGWeight, self).__init__()
        self.input_a_l = input_a_l
        self.input_h_l = input_h_l
        self.l_out = l_out
        self.out_dim = 2 * l_out + 1

        # Precompute and store all valid CG combinations with coefficients
        self.valid_combos = []
        self.cg_coefficients = []

        combo_idx = 0
        for a_l_idx, a_l_in in enumerate(input_a_l):
            for h_l_idx, h_l_in in enumerate(input_h_l):
                try:
                    validate_triangle_inequality(a_l_in, h_l_in, l_out)

                    # Get CG coefficients
                    cg_coeffs = CGCoefficientsCache.get_coefficients(
                        a_l_in, h_l_in, l_out, torch.device("cpu")
                    )

                    if cg_coeffs is not None and cg_coeffs.abs().sum() > 1e-10:
                        self.valid_combos.append(
                            (combo_idx, a_l_idx, h_l_idx, a_l_in, h_l_in)
                        )
                        self.cg_coefficients.append(cg_coeffs)
                        combo_idx += 1
                except (ValueError, Exception):
                    continue

        if len(self.valid_combos) == 0:
            raise ValueError(
                f"No valid CG combinations possible for input_a_l={input_a_l}, "
                f"input_h_l={input_h_l}, l_out={l_out}. All combinations violate triangle inequality."
            )

        # Initialize weights parameter - only for valid combinations
        self.weight = torch.nn.parameter.Parameter(
            torch.zeros(len(self.valid_combos), dtype=torch.float32)
        )

        # Register CG coefficients as buffers (non-trainable parameters that move with model)
        for i, cg_coeffs in enumerate(self.cg_coefficients):
            self.register_buffer(f"cg_coeffs_{i}", cg_coeffs)

        # Apply initialization
        fan_in = len(self.valid_combos)
        fan_out = self.out_dim

        if init_method == "xavier":
            xavier_init_steerable(self.weight, fan_in, fan_out)
        elif init_method == "xavier_uniform":
            xavier_uniform_init_steerable(self.weight, fan_in, fan_out)
        elif init_method == "constant":
            with torch.no_grad():
                self.weight.fill_(0.1)
        elif init_method == "uniform":
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.1
            with torch.no_grad():
                self.weight.uniform_(-bound, bound)
        elif init_method == "kaiming":
            std = math.sqrt(2.0 / fan_in) if fan_in > 0 else 0.1
            with torch.no_grad():
                self.weight.normal_(0, std)

    def forward(self, input_a_batch, input_h_batch):
        """
        Ultra-fast vectorized forward pass using precomputed CG coefficients
        """
        if len(input_a_batch) == 0 or len(input_h_batch) == 0:
            return torch.zeros(
                0, self.out_dim, dtype=torch.float32, device=self.weight.device
            )

        num_edges = input_a_batch[0].shape[0]
        result = torch.zeros(
            num_edges, self.out_dim, dtype=torch.float32, device=self.weight.device
        )

        # Process all valid combinations efficiently
        for i, (combo_idx, a_l_idx, h_l_idx, a_l_in, h_l_in) in enumerate(
            self.valid_combos
        ):
            # Get precomputed CG coefficients (already on correct device)
            cg_coeffs = getattr(self, f"cg_coeffs_{i}")

            # Safety checks with early exit
            if a_l_idx >= len(input_a_batch) or h_l_idx >= len(input_h_batch):
                continue

            # Get input tensors
            a_tensor = input_a_batch[a_l_idx]  # [E, 2*a_l_in+1]
            h_tensor = input_h_batch[h_l_idx]  # [E, 2*h_l_in+1]

            # Apply CG product: einsum is very optimized in PyTorch
            temp = torch.einsum("ijk,ei,ej->ek", cg_coeffs, a_tensor, h_tensor)

            # Weighted accumulation - use in-place operation for speed
            result.add_(temp, alpha=self.weight[i].item())

        return result


class HiddenHLayer(torch.nn.Module):
    def __init__(self, input_a_l, input_h_l, h_l_out, init_method="xavier"):
        super(HiddenHLayer, self).__init__()
        # Use ModuleList instead of regular Python list to properly register parameters
        self.cg_weight = torch.nn.ModuleList(
            [CGWeight(input_a_l, input_h_l, l_out, init_method) for l_out in h_l_out]
        )
        self.h_l_out = h_l_out

    def forward(self, input_a_batch, input_h_batch):
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
            output = self.cg_weight[i].forward(input_a_batch, input_h_batch)
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
            input_a_l, self.intermediate_dims, self.intermediate_dims, init_method
        )
        self.hidden_h_layer_3 = HiddenHLayer(
            input_a_l, self.intermediate_dims, h_l_out, init_method
        )
        self.h_l_out = h_l_out

    def forward(self, input_a_batch, input_h_batch):
        """
        Vectorized forward pass for batch of edges

        Args:
            input_a_batch: list of [E, irrep_dim] tensors for edge features
            input_h_batch: list of [E, irrep_dim] tensors for source node features

        Returns:
            list of [E, irrep_dim] tensors for output messages
        """
        # First layer outputs a list of tensors
        h1_batch = self.hidden_h_layer_1.forward(input_a_batch, input_h_batch)

        # Second layer with h1 list
        h2_batch = self.hidden_h_layer_2.forward(input_a_batch, h1_batch)

        # Third layer for final output
        h3_batch = self.hidden_h_layer_3.forward(input_a_batch, h2_batch)

        return h3_batch


class MessageLayer(nn.Module):
    """Ultra-fast message passing layer - proper equivariance with GPU optimization"""

    def __init__(self, max_sh_degree, irrep_dims, hidden_dim):
        super().__init__()
        self.max_sh_degree = max_sh_degree
        self.irrep_dims = irrep_dims  # [1, 3] for [scalar, vector]
        self.total_dim = sum(irrep_dims)  # 4

        # Learnable weights for each valid CG combination
        self.weights = nn.ParameterDict()

        # Precompute CG coefficients and find valid combinations
        self._precompute_layer_cg_coefficients()

    def _precompute_layer_cg_coefficients(self):
        """Precompute and store CG coefficients for this layer"""

        # Map irrep indices to actual l quantum numbers
        # irrep_dims = [1, 3] corresponds to l = [0, 1]
        irrep_idx_to_l = []
        for irrep_idx, dim in enumerate(self.irrep_dims):
            # For irrep_dims[i], the l value is such that 2*l+1 = dim
            l = (dim - 1) // 2
            irrep_idx_to_l.append(l)

        # Store all CG coefficients that we might need
        for edge_sh_degree in range(self.max_sh_degree + 1):
            for node_irrep_idx in range(len(self.irrep_dims)):
                for out_irrep_idx in range(len(self.irrep_dims)):
                    try:
                        # Convert indices to actual l quantum numbers for triangle inequality
                        node_l = irrep_idx_to_l[node_irrep_idx]  # 0 -> l=0, 1 -> l=1
                        out_l = irrep_idx_to_l[out_irrep_idx]  # 0 -> l=0, 1 -> l=1

                        # Now validate with actual l values: edge_sh_degree ⊗ node_l → out_l
                        validate_triangle_inequality(edge_sh_degree, node_l, out_l)

                        # Use actual l values for CG computation
                        cg = _wigner._so3_clebsch_gordan(
                            edge_sh_degree, node_l, out_l
                        ).float()
                        if cg.abs().sum() > 1e-10:
                            key = (
                                f"cg_{edge_sh_degree}_{node_irrep_idx}_{out_irrep_idx}"
                            )
                            self.register_buffer(key, cg)
                    except:
                        continue

        # Now find valid combinations (after CG coefficients are registered)
        self.valid_combinations = []
        for edge_sh_degree in range(self.max_sh_degree + 1):
            for node_irrep_idx in range(len(self.irrep_dims)):
                for out_irrep_idx in range(len(self.irrep_dims)):
                    cg_key = f"cg_{edge_sh_degree}_{node_irrep_idx}_{out_irrep_idx}"
                    # Check if we successfully registered this CG coefficient
                    for name, _ in self.named_buffers():
                        if name == cg_key:
                            self.valid_combinations.append(
                                (edge_sh_degree, node_irrep_idx, out_irrep_idx)
                            )
                            # Create learnable weight
                            weight_key = (
                                f"w_{edge_sh_degree}_{node_irrep_idx}_{out_irrep_idx}"
                            )
                            self.weights[weight_key] = nn.Parameter(
                                torch.randn(1) * 0.1
                            )
                            break

    def forward(self, node_irreps, edge_index, sh_edge_features):
        """Ultra-fast message passing with proper equivariance"""
        device = node_irreps.device
        num_nodes = node_irreps.shape[0]
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            return node_irreps

        # Get source and target indices (these are actual graph structure indices)
        source_idx = edge_index[0]  # [E] - actual source nodes in graph
        target_idx = edge_index[1]  # [E] - actual target nodes in graph

        # Gather source node features
        source_features = node_irreps[source_idx]  # [E, 4]

        # Split source features by irrep type for proper CG operations
        source_irreps = []
        start_idx = 0
        for dim in self.irrep_dims:
            end_idx = start_idx + dim
            source_irreps.append(source_features[:, start_idx:end_idx])
            start_idx = end_idx

        # Initialize output messages by irrep type
        message_irreps = []
        for dim in self.irrep_dims:
            message_irreps.append(torch.zeros(num_edges, dim, device=device))

        # Process valid CG combinations with proper tensor products
        # Note: edge_sh_degree, node_irrep_idx, out_irrep_idx refer to FEATURE TYPES, not graph structure
        for edge_sh_degree, node_irrep_idx, out_irrep_idx in self.valid_combinations:
            weight_key = f"w_{edge_sh_degree}_{node_irrep_idx}_{out_irrep_idx}"
            cg_key = f"cg_{edge_sh_degree}_{node_irrep_idx}_{out_irrep_idx}"

            weight = self.weights[weight_key]
            cg_coeffs = getattr(self, cg_key)

            if edge_sh_degree >= len(sh_edge_features):
                continue

            edge_feat = sh_edge_features[edge_sh_degree]  # [E, 2*edge_sh_degree+1]
            node_feat = source_irreps[node_irrep_idx]  # [E, 2*node_irrep_idx+1]

            # Proper CG tensor product: cg[i,j,k] * edge[i] * node[j] -> out[k]
            msg = torch.einsum(
                "ijk,ei,ej->ek", cg_coeffs, edge_feat, node_feat
            )  # [E, 2*out_irrep_idx+1]

            # Apply weight and accumulate in the correct irrep
            message_irreps[out_irrep_idx] += weight * msg

        # Aggregate messages by irrep type using efficient scatter operations
        aggregated_irreps = []
        for irrep_idx in range(len(self.irrep_dims)):
            dim = self.irrep_dims[irrep_idx]
            agg_tensor = torch.zeros(num_nodes, dim, device=device)
            agg_tensor.index_add_(0, target_idx, message_irreps[irrep_idx])
            aggregated_irreps.append(agg_tensor)

        # Normalize by message count
        counts = torch.zeros(num_nodes, device=device)
        counts.index_add_(0, target_idx, torch.ones(num_edges, device=device))
        counts = torch.clamp(counts, min=1.0)

        # Normalize each irrep separately (maintaining equivariance)
        for irrep_idx in range(len(self.irrep_dims)):
            aggregated_irreps[irrep_idx] = aggregated_irreps[
                irrep_idx
            ] / counts.unsqueeze(-1)

        # Update node features WITHOUT residual connections (they break equivariance!)
        # Simply replace with the aggregated messages
        updated_irreps = torch.cat(aggregated_irreps, dim=1)  # [N, total_dim]

        return updated_irreps


class EquivariantGNN(torch.nn.Module):
    """
    Ultra-fast GPU-optimized equivariant GNN - mathematically correct version

    Key features:
    1. Proper CG tensor products for exact equivariance
    2. GPU-optimized tensor operations
    3. No operations that break equivariance (like residual connections)
    4. Precomputed CG coefficients as buffers
    """

    def __init__(
        self,
        input_dim=3,
        hidden_dim=16,
        message_passing_steps=3,
        final_mlp_dims=[64, 32],
        max_sh_degree=1,
        init_method="xavier",
        seed=42,
        debug=False,
    ):
        super().__init__()
        torch.manual_seed(seed)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_sh_degree = max_sh_degree
        self.debug = debug
        self.message_passing_steps = message_passing_steps

        # Fixed irrep structure for center of mass: [0, 1] (scalar + vector)
        self.irrep_dims = [1, 3]  # l=0: 1 dim, l=1: 3 dims
        self.total_irrep_dim = sum(self.irrep_dims)  # 4 total

        # Input encoding: 3D -> irrep representation [scalar, vector]
        # FIXED: Use proper equivariant encoders

        # Scalar encoder: l=0 (rotation-invariant)
        # Use radial distance as a true scalar invariant
        # No trainable parameters needed - this is a fixed mathematical operation

        # Vector encoder: l=1 (rotation-equivariant)
        # Constrain to be c*I to maintain proper vector representation
        self.vector_scale = nn.Parameter(torch.tensor(0.1))  # Single learnable scalar

        # Precompute ALL CG coefficients we need and store as buffers
        self._precompute_cg_coefficients()

        # Message passing layers - each reduces and then expands features
        self.message_layers = nn.ModuleList(
            [
                MessageLayer(
                    max_sh_degree=max_sh_degree,
                    irrep_dims=self.irrep_dims,
                    hidden_dim=hidden_dim,
                )
                for _ in range(message_passing_steps)
            ]
        )

        # Final MLP: invariant features -> scalar weights
        final_layers = []
        prev_dim = 2  # l=0 scalar + l=1 invariant scalar
        for dim in final_mlp_dims:
            final_layers.append(nn.Linear(prev_dim, dim))
            final_layers.append(nn.ReLU())
            prev_dim = dim
        final_layers.append(nn.Linear(prev_dim, 1))
        self.final_mlp = nn.Sequential(*final_layers)

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better training stability"""
        # No need to initialize encoders anymore - they use proper mathematical operations
        # The scalar encoder is now a fixed norm operation
        # The vector encoder is now a single learnable scale parameter (already initialized above)
        pass

    def _precompute_cg_coefficients(self):
        """Precompute and register all CG coefficients as buffers"""
        # CG for invariant features: l=1 ⊗ l=1 -> l=0
        cg_l1_l1_l0 = _wigner._so3_clebsch_gordan(1, 1, 0).float()  # [3, 3, 1]
        self.register_buffer("cg_l1_l1_l0", cg_l1_l1_l0)

    def forward(self, x, edge_index, edge_attr, batch=None, node_pos=None):
        """Ultra-fast forward pass with proper equivariance"""
        if self.debug:
            start_time = time.time()

        device = x.device
        num_nodes = x.shape[0]

        # Use positions
        if node_pos is None:
            node_pos = x

        # Encode coordinates directly (no centering) to maintain equivariance
        # The translation invariance should come from the message passing structure,
        # not from preprocessing

        # FIXED: Use proper equivariant encoders
        # Scalar features (l=0): rotation-invariant norm
        scalar_features = torch.norm(
            x, dim=-1, keepdim=True
        )  # [N, 1] - truly invariant

        # Vector features (l=1): scaled identity transformation
        vector_features = self.vector_scale * x  # [N, 3] - proper vector representation

        # Combine into single tensor: [N, 4] where first 1 is scalar, next 3 is vector
        node_irreps = torch.cat([scalar_features, vector_features], dim=1)  # [N, 4]

        if self.debug:
            embed_time = time.time() - start_time
            mp_start = time.time()

        # Split edge attributes to irrep format efficiently
        sh_edge_features = self._split_sh_features_tensor(edge_attr)

        # Message passing with proper equivariance
        for i, layer in enumerate(self.message_layers):
            if self.debug:
                layer_start = time.time()
            node_irreps = layer(node_irreps, edge_index, sh_edge_features)
            if self.debug:
                print(f"[DEBUG] Corrected Layer {i}: {time.time() - layer_start:.3f}s")

        if self.debug:
            mp_time = time.time() - mp_start
            final_start = time.time()

        # Extract invariant features efficiently
        l0_features = node_irreps[:, 0:1]  # [N, 1] - scalar
        l1_features = node_irreps[:, 1:4]  # [N, 3] - vector

        # Compute l1 invariants using precomputed CG coefficients
        l1_invariants = torch.einsum(
            "ijk,ni,nj->nk", self.cg_l1_l1_l0, l1_features, l1_features
        )  # [N, 1]

        # Combine invariant features
        invariant_features = torch.cat([l0_features, l1_invariants], dim=1)  # [N, 2]

        # Generate weights and compute COM
        weights = torch.softmax(
            self.final_mlp(invariant_features).squeeze(-1), dim=0
        )  # [N]

        # Handle batched case properly
        if batch is not None:
            # For batched processing, compute COM per graph
            unique_batches = torch.unique(batch)
            com_predictions = []
            for b in unique_batches:
                mask = batch == b
                batch_weights = weights[mask]
                batch_positions = node_pos[mask]
                batch_weights = torch.softmax(
                    batch_weights, dim=0
                )  # Re-normalize within batch
                batch_com = torch.sum(
                    batch_weights.unsqueeze(-1) * batch_positions, dim=0
                )
                com_predictions.append(batch_com)
            com_prediction = torch.stack(com_predictions, dim=0)  # [B, 3]
        else:
            com_prediction = torch.sum(
                weights.unsqueeze(-1) * node_pos, dim=0, keepdim=True
            )  # [1, 3]

        if self.debug:
            final_time = time.time() - final_start
            total_time = time.time() - start_time
            print(
                f"[DEBUG] Corrected - Embed: {embed_time:.3f}s, MP: {mp_time:.3f}s, Final: {final_time:.3f}s"
            )
            print(f"[DEBUG] Corrected Total: {total_time:.3f}s")

        return com_prediction

    def _split_sh_features_tensor(self, edge_attr):
        """Split SH features efficiently keeping as tensors"""
        features = []
        start_idx = 0
        for l in range(self.max_sh_degree + 1):
            dim_l = 2 * l + 1
            end_idx = start_idx + dim_l
            features.append(edge_attr[:, start_idx:end_idx].contiguous())
            start_idx = end_idx
        return features
