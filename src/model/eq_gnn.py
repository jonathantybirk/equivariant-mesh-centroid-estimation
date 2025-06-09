import torch
import torch.nn as nn
from e3nn.o3 import _wigner
import math
import time
from ..base_model import BaseModel


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
                # Always compute on CPU since we register as buffers that move with model
                cls._cache[key] = _wigner._so3_clebsch_gordan(l1, l2, l3).float()
            except Exception:
                cls._cache[key] = None

        cg_coeffs = cls._cache[key]
        if cg_coeffs is not None:
            # If device is None, return on CPU (will be moved via register_buffer)
            if device is not None:
                return cg_coeffs.to(device)
            else:
                return cg_coeffs
        return None


class MessageLayer(nn.Module):
    """Ultra-fast message passing layer - proper equivariance with GPU optimization"""

    def __init__(self, max_sh_degree, irrep_dims, multiplicity=1):
        super().__init__()
        self.max_sh_degree = max_sh_degree
        self.irrep_dims = (
            irrep_dims  # [1*mult, 3*mult] for [scalar, vector] with multiplicity
        )
        self.base_irrep_dims = [1, 3]  # Base dimensions without multiplicity
        self.multiplicity = multiplicity
        self.total_dim = sum(irrep_dims)

        # Learnable weights for each valid CG combination AND multiplicity channel
        self.weights = nn.ParameterDict()

        # Precompute CG coefficients and find valid combinations
        self._precompute_layer_cg_coefficients()

    def _precompute_layer_cg_coefficients(self):
        """Precompute and store CG coefficients for this layer"""

        # Map irrep indices to actual l quantum numbers
        # Use base dimensions for CG coefficient computation
        irrep_idx_to_l = []
        for irrep_idx, dim in enumerate(self.base_irrep_dims):
            # For base_irrep_dims[i], the l value is such that 2*l+1 = dim
            l = (dim - 1) // 2
            irrep_idx_to_l.append(l)

        # Store all CG coefficients that we might need
        for edge_sh_degree in range(self.max_sh_degree + 1):
            for node_irrep_idx in range(len(self.base_irrep_dims)):
                for out_irrep_idx in range(len(self.base_irrep_dims)):
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
            for node_irrep_idx in range(len(self.base_irrep_dims)):
                for out_irrep_idx in range(len(self.base_irrep_dims)):
                    cg_key = f"cg_{edge_sh_degree}_{node_irrep_idx}_{out_irrep_idx}"
                    # Check if we successfully registered this CG coefficient
                    for name, _ in self.named_buffers():
                        if name == cg_key:
                            self.valid_combinations.append(
                                (edge_sh_degree, node_irrep_idx, out_irrep_idx)
                            )
                            # Create learnable weights for each multiplicity channel
                            for in_channel in range(self.multiplicity):
                                for out_channel in range(self.multiplicity):
                                    weight_key = f"w_{edge_sh_degree}_{node_irrep_idx}_{out_irrep_idx}_{in_channel}_{out_channel}"
                                    self.weights[weight_key] = nn.Parameter(
                                        torch.randn(1) * 1.0
                                    )
                            break

    def forward(self, node_irreps, edge_index, sh_edge_features):
        """Ultra-fast message passing with proper equivariance and multiplicity support"""
        device = node_irreps.device
        num_nodes = node_irreps.shape[0]
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            return node_irreps

        # Get source and target indices (these are actual graph structure indices)
        source_idx = edge_index[0]  # [E] - actual source nodes in graph
        target_idx = edge_index[1]  # [E] - actual target nodes in graph

        # Gather source node features
        source_features = node_irreps[source_idx]  # [E, total_dim]

        # Split source features by irrep type AND multiplicity channel
        # For multiplicity=2: [scalar_ch0, scalar_ch1, vector_ch0, vector_ch1]
        source_irrep_channels = []
        start_idx = 0
        for irrep_idx, base_dim in enumerate(self.base_irrep_dims):
            # Each irrep type has multiplicity channels
            for channel in range(self.multiplicity):
                end_idx = start_idx + base_dim
                source_irrep_channels.append(source_features[:, start_idx:end_idx])
                start_idx = end_idx

        # Initialize output messages by irrep type and channel
        message_irrep_channels = []
        for irrep_idx, base_dim in enumerate(self.base_irrep_dims):
            for channel in range(self.multiplicity):
                message_irrep_channels.append(
                    torch.zeros(num_edges, base_dim, device=device)
                )

        # Process valid CG combinations with multiplicity
        for edge_sh_degree, node_irrep_idx, out_irrep_idx in self.valid_combinations:
            cg_key = f"cg_{edge_sh_degree}_{node_irrep_idx}_{out_irrep_idx}"
            cg_coeffs = getattr(self, cg_key)

            if edge_sh_degree >= len(sh_edge_features):
                continue

            edge_feat = sh_edge_features[edge_sh_degree]  # [E, 2*edge_sh_degree+1]

            # Process all channel combinations
            for in_channel in range(self.multiplicity):
                for out_channel in range(self.multiplicity):
                    weight_key = f"w_{edge_sh_degree}_{node_irrep_idx}_{out_irrep_idx}_{in_channel}_{out_channel}"
                    weight = self.weights[weight_key]

                    # Get input channel: node_irrep_idx determines base irrep, in_channel determines which channel
                    input_channel_idx = node_irrep_idx * self.multiplicity + in_channel
                    node_feat = source_irrep_channels[
                        input_channel_idx
                    ]  # [E, base_dim]

                    # Proper CG tensor product: cg[i,j,k] * edge[i] * node[j] -> out[k]
                    msg = torch.einsum(
                        "ijk,ei,ej->ek", cg_coeffs, edge_feat, node_feat
                    )  # [E, base_dim]

                    # Add to output channel: out_irrep_idx determines base irrep, out_channel determines which channel
                    output_channel_idx = out_irrep_idx * self.multiplicity + out_channel
                    message_irrep_channels[output_channel_idx] += weight * msg

        # Aggregate messages by channel using efficient scatter operations
        aggregated_channels = []
        for channel_idx in range(len(message_irrep_channels)):
            base_dim = message_irrep_channels[channel_idx].shape[1]
            agg_tensor = torch.zeros(num_nodes, base_dim, device=device)
            agg_tensor.index_add_(0, target_idx, message_irrep_channels[channel_idx])
            aggregated_channels.append(agg_tensor)

        # Concatenate all channels back into irrep format
        updated_irreps = torch.cat(aggregated_channels, dim=1)  # [N, total_dim]

        return updated_irreps


class EquivariantGNN(BaseModel):
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
        message_passing_steps=3,
        final_mlp_dims=[64, 32],
        max_sh_degree=1,
        init_method="xavier",
        seed=42,
        debug=False,
        lr=None,
        weight_decay=None,
        multiplicity=2,
        dropout=0.2,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        torch.manual_seed(seed)

        self.max_sh_degree = max_sh_degree
        self.debug = debug
        self.message_passing_steps = message_passing_steps
        self.multiplicity = multiplicity
        self.dropout = dropout

        # Fixed irrep structure for center of mass: [0, 1] (scalar + vector)
        # With multiplicity, each irrep type has multiple channels
        self.base_irrep_dims = [1, 3]  # l=0: 1 dim, l=1: 3 dims
        self.irrep_dims = [
            dim * multiplicity for dim in self.base_irrep_dims
        ]  # Account for multiplicity
        self.total_irrep_dim = sum(self.irrep_dims)  # Total with multiplicity

        # Input encoding: Node features start with distance (l=0) and zeros (l=1)
        # The l=1 features will be built up through message passing from edge spherical harmonics

        # Precompute ALL CG coefficients we need and store as buffers
        self._precompute_cg_coefficients()

        # Message passing layers - each reduces and then expands features
        self.message_layers = nn.ModuleList(
            [
                MessageLayer(
                    max_sh_degree=max_sh_degree,
                    irrep_dims=self.irrep_dims,
                    multiplicity=multiplicity,
                )
                for _ in range(self.message_passing_steps)
            ]
        )

        # Final MLP with proper regularization: invariant features -> scalar weights
        final_layers = []
        prev_dim = 2  # l=0 scalar + l=1 invariant scalar
        for i, dim in enumerate(final_mlp_dims):
            final_layers.append(nn.Linear(prev_dim, dim))
            final_layers.append(
                nn.LayerNorm(dim)
            )  # Layer normalization for stable training
            final_layers.append(nn.ReLU())
            if dropout > 0:
                final_layers.append(nn.Dropout(dropout))  # Dropout for regularization
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

        # BRILLIANT APPROACH: Trivial initial node features with multiplicity
        # Initialize all channels for each irrep type
        initial_features = []

        # l=0 (scalar): Set to 1 for all channels
        for channel in range(self.multiplicity):
            scalar_channel = torch.ones(
                num_nodes, 1, device=device
            )  # [N, 1] - all same
            initial_features.append(scalar_channel)

        # l=1 (vector): Initialize as zeros for all channels
        for channel in range(self.multiplicity):
            vector_channel = torch.zeros(
                num_nodes, 3, device=device
            )  # [N, 3] - start with zeros
            initial_features.append(vector_channel)

        # Combine into single tensor: [N, total_irrep_dim]
        node_irreps = torch.cat(initial_features, dim=1)  # [N, 2+6] for mult=2

        if self.debug:
            embed_time = time.time() - start_time
            mp_start = time.time()

        # Split edge attributes to irrep format efficiently
        sh_edge_features = self._split_sh_features_tensor(edge_attr)

        # Message passing with proper equivariance - this will build up the vector features
        for i, layer in enumerate(self.message_layers):
            if self.debug:
                layer_start = time.time()
            node_irreps = layer(node_irreps, edge_index, sh_edge_features)
            if self.debug:
                print(f"[DEBUG] Layer {i}: {time.time() - layer_start:.3f}s")

        if self.debug:
            mp_time = time.time() - mp_start
            final_start = time.time()

        # Extract invariant features efficiently - handle multiplicity
        # For multiplicity=2: [scalar_ch0, scalar_ch1, vector_ch0, vector_ch1]

        # Extract all scalar channels and combine (simple average)
        scalar_channels = []
        for channel in range(self.multiplicity):
            start_idx = channel * 1  # 1 = base scalar dim
            end_idx = start_idx + 1
            scalar_channels.append(node_irreps[:, start_idx:end_idx])
        l0_features = torch.mean(
            torch.cat(scalar_channels, dim=1), dim=1, keepdim=True
        )  # [N, 1]

        # Extract all vector channels and combine (simple average)
        vector_channels = []
        vector_start_offset = self.multiplicity * 1  # Skip all scalar channels
        for channel in range(self.multiplicity):
            start_idx = vector_start_offset + channel * 3  # 3 = base vector dim
            end_idx = start_idx + 3
            vector_channels.append(node_irreps[:, start_idx:end_idx])
        l1_features = torch.mean(torch.stack(vector_channels, dim=0), dim=0)  # [N, 3]

        # Compute l1 invariants using precomputed CG coefficients
        l1_invariants = torch.einsum(
            "ijk,ni,nj->nk", self.cg_l1_l1_l0, l1_features, l1_features
        )  # [N, 1]

        # Combine invariant features
        invariant_features = torch.cat([l0_features, l1_invariants], dim=1)  # [N, 2]

        # Compute raw logits (no softmax yet)
        raw_logits = self.final_mlp(invariant_features).squeeze(-1)  # [N]

        # DISPLACEMENT FROM MEAN APPROACH
        if batch is not None:
            # For batched processing, compute displacement per graph
            unique_batches = torch.unique(batch)
            com_predictions = []
            for i, b in enumerate(unique_batches):
                mask = batch == b
                batch_logits = raw_logits[mask]
                batch_positions = node_pos[mask]

                # Compute geometric mean of this graph
                geometric_mean = batch_positions.mean(dim=0)  # [3]

                # Apply softmax ONLY within each graph to get attention weights
                batch_weights = torch.softmax(batch_logits, dim=0)  # [N_batch]

                # Predict displacement from geometric mean
                displacement = (
                    torch.sum(batch_weights.unsqueeze(-1) * batch_positions, dim=0)
                    - geometric_mean
                )  # [3]

                # Final COM = geometric_mean + displacement
                batch_com = geometric_mean + displacement
                com_predictions.append(batch_com)
            com_prediction = torch.stack(com_predictions, dim=0)  # [B, 3]
        else:
            # Single graph case
            geometric_mean = node_pos.mean(dim=0)  # [3]
            weights = torch.softmax(raw_logits, dim=0)

            # Predict displacement from geometric mean
            displacement = (
                torch.sum(weights.unsqueeze(-1) * node_pos, dim=0) - geometric_mean
            )  # [3]

            # Final COM = geometric_mean + displacement
            com_prediction = (geometric_mean + displacement).unsqueeze(0)  # [1, 3]

        if self.debug:
            final_time = time.time() - final_start
            total_time = time.time() - start_time
            print(
                f"[DEBUG] Embed: {embed_time:.3f}s, MP: {mp_time:.3f}s, Final: {final_time:.3f}s"
            )
            print(f"[DEBUG] Total: {total_time:.3f}s")

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
