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


class MultiCGLayer(nn.Module):
    """
    Layer that applies multiple CG tensor products in sequence to create richer messages
    """

    def __init__(
        self, max_sh_degree, base_l_values, multiplicity, num_cg_layers, debug=False
    ):
        super().__init__()
        self.max_sh_degree = max_sh_degree
        self.base_l_values = base_l_values
        self.multiplicity = multiplicity
        self.num_cg_layers = num_cg_layers
        self.debug = debug

        # Create multiple CG layers
        self.cg_layers = nn.ModuleList()
        for layer_idx in range(self.num_cg_layers):
            layer_weights = nn.ParameterDict()
            # For each layer, precompute CG coefficients and create weights
            self._setup_cg_layer(layer_idx, layer_weights)
            self.cg_layers.append(layer_weights)

    def _setup_cg_layer(self, layer_idx, layer_weights):
        """Setup CG coefficients and weights for one layer"""
        # Store valid combinations for this layer
        valid_combinations = []

        if self.debug:
            print(f"\n[DEBUG] Setting up CG Layer {layer_idx}")
            print(f"[DEBUG] max_sh_degree: {self.max_sh_degree}")
            print(f"[DEBUG] base_l_values: {self.base_l_values}")
            print(f"[DEBUG] multiplicity: {self.multiplicity}")

        # Precompute CG coefficients
        for edge_sh_degree in range(self.max_sh_degree + 1):
            for node_irrep_idx in range(len(self.base_l_values)):
                for out_irrep_idx in range(len(self.base_l_values)):
                    try:
                        node_l = self.base_l_values[node_irrep_idx]
                        out_l = self.base_l_values[out_irrep_idx]

                        validate_triangle_inequality(edge_sh_degree, node_l, out_l)

                        cg = _wigner._so3_clebsch_gordan(
                            edge_sh_degree, node_l, out_l
                        ).float()

                        if cg.abs().sum() > 1e-10:
                            # Register CG coefficients as buffer
                            cg_key = f"cg_{layer_idx}_{edge_sh_degree}_{node_irrep_idx}_{out_irrep_idx}"
                            self.register_buffer(cg_key, cg)

                            valid_combinations.append(
                                (edge_sh_degree, node_irrep_idx, out_irrep_idx)
                            )

                            if self.debug:
                                print(
                                    f"[DEBUG] Valid CG: l_edge={edge_sh_degree} ⊗ l_node={node_l} → l_out={out_l} | CG shape: {cg.shape}"
                                )

                            # Create learnable weights for each multiplicity channel
                            for in_channel in range(self.multiplicity):
                                for out_channel in range(self.multiplicity):
                                    weight_key = f"w_{edge_sh_degree}_{node_irrep_idx}_{out_irrep_idx}_{in_channel}_{out_channel}"
                                    layer_weights[weight_key] = nn.Parameter(
                                        torch.randn(1) * 0.1
                                    )
                    except Exception as e:
                        if self.debug:
                            node_l = (
                                self.base_l_values[node_irrep_idx]
                                if node_irrep_idx < len(self.base_l_values)
                                else "?"
                            )
                            out_l = (
                                self.base_l_values[out_irrep_idx]
                                if out_irrep_idx < len(self.base_l_values)
                                else "?"
                            )
                            print(
                                f"[DEBUG] Invalid CG: l_edge={edge_sh_degree} ⊗ l_node={node_l} → l_out={out_l} | Error: {str(e)}"
                            )
                        continue

        # Store valid combinations for this layer
        setattr(self, f"valid_combinations_{layer_idx}", valid_combinations)

        if self.debug:
            total_weights = (
                len(valid_combinations) * self.multiplicity * self.multiplicity
            )
            print(
                f"[DEBUG] Layer {layer_idx}: {len(valid_combinations)} valid CG combinations"
            )
            print(f"[DEBUG] Layer {layer_idx}: {total_weights} total learnable weights")
            print(f"[DEBUG] Valid combinations: {valid_combinations}")
            print()  # Empty line for readability

    def forward(self, node_irreps, edge_index, sh_edge_features):
        """Apply multiple CG operations in sequence"""
        device = node_irreps.device
        num_nodes = node_irreps.shape[0]
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            return node_irreps

        # Keep track of intermediate results
        current_features = node_irreps

        # Apply each CG layer sequentially
        for layer_idx in range(self.num_cg_layers):
            current_features = self._apply_cg_layer(
                layer_idx, current_features, edge_index, sh_edge_features
            )

        return current_features

    def _apply_cg_layer(self, layer_idx, node_irreps, edge_index, sh_edge_features):
        """Apply one CG layer"""
        device = node_irreps.device
        num_nodes = node_irreps.shape[0]
        num_edges = edge_index.shape[1]

        source_idx = edge_index[0]
        target_idx = edge_index[1]
        source_features = node_irreps[source_idx]

        # Split source features by irrep type and multiplicity channel
        source_irrep_channels = []
        start_idx = 0
        for irrep_idx, l_value in enumerate(self.base_l_values):
            base_dim = 2 * l_value + 1  # Compute dimension from l value
            for channel in range(self.multiplicity):
                end_idx = start_idx + base_dim
                source_irrep_channels.append(source_features[:, start_idx:end_idx])
                start_idx = end_idx

        # Initialize output messages
        message_irrep_channels = []
        for irrep_idx, l_value in enumerate(self.base_l_values):
            base_dim = 2 * l_value + 1  # Compute dimension from l value
            for channel in range(self.multiplicity):
                message_irrep_channels.append(
                    torch.zeros(num_edges, base_dim, device=device)
                )

        # Get valid combinations and weights for this layer
        valid_combinations = getattr(self, f"valid_combinations_{layer_idx}")
        layer_weights = self.cg_layers[layer_idx]

        # Process valid CG combinations
        for edge_sh_degree, node_irrep_idx, out_irrep_idx in valid_combinations:
            cg_key = f"cg_{layer_idx}_{edge_sh_degree}_{node_irrep_idx}_{out_irrep_idx}"
            cg_coeffs = getattr(self, cg_key)

            if edge_sh_degree >= len(sh_edge_features):
                continue

            edge_feat = sh_edge_features[edge_sh_degree]

            # Process all channel combinations
            for in_channel in range(self.multiplicity):
                for out_channel in range(self.multiplicity):
                    weight_key = f"w_{edge_sh_degree}_{node_irrep_idx}_{out_irrep_idx}_{in_channel}_{out_channel}"
                    weight = layer_weights[weight_key]

                    input_channel_idx = node_irrep_idx * self.multiplicity + in_channel
                    node_feat = source_irrep_channels[input_channel_idx]

                    # CG tensor product
                    msg = torch.einsum("ijk,ei,ej->ek", cg_coeffs, edge_feat, node_feat)

                    output_channel_idx = out_irrep_idx * self.multiplicity + out_channel
                    message_irrep_channels[output_channel_idx] += weight * msg

        # Aggregate messages
        aggregated_channels = []
        for channel_idx in range(len(message_irrep_channels)):
            base_dim = message_irrep_channels[channel_idx].shape[1]
            agg_tensor = torch.zeros(num_nodes, base_dim, device=device)
            agg_tensor.index_add_(0, target_idx, message_irrep_channels[channel_idx])
            aggregated_channels.append(agg_tensor)

        updated_irreps = torch.cat(aggregated_channels, dim=1)
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
        message_passing_steps=4,
        final_mlp_dims=[64, 32],
        max_sh_degree=4,
        init_method="xavier",
        seed=42,
        debug=False,
        lr=None,
        weight_decay=None,
        multiplicity=2,
        dropout=0.2,
        num_cg_layers=4,
        base_l_values=[0, 1, 2, 3, 4],
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        torch.manual_seed(seed)

        self.max_sh_degree = max_sh_degree
        self.debug = debug
        self.message_passing_steps = message_passing_steps
        self.multiplicity = multiplicity
        self.dropout = dropout
        self.num_cg_layers = num_cg_layers
        self.base_l_values = base_l_values

        # Precompute ALL CG coefficients we need and store as buffers
        self._precompute_cg_coefficients()

        # Message passing layers using MultiCGLayer
        self.message_layers = nn.ModuleList(
            [
                MultiCGLayer(
                    max_sh_degree=self.max_sh_degree,
                    base_l_values=self.base_l_values,
                    multiplicity=self.multiplicity,
                    num_cg_layers=self.num_cg_layers,
                    debug=self.debug,
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

        # Initialize node features for each irrep type and multiplicity channel
        initial_features = []

        for l_value in self.base_l_values:
            irrep_dim = 2 * l_value + 1  # Compute dimension from l value
            for channel in range(self.multiplicity):
                if l_value == 0:  # Scalar features - initialize to 1
                    irrep_channel = torch.ones(num_nodes, irrep_dim, device=device)
                else:  # Higher order features - initialize to zeros
                    irrep_channel = torch.zeros(num_nodes, irrep_dim, device=device)
                initial_features.append(irrep_channel)

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
        invariant_features_list = []

        # Extract features by l value and compute invariants
        start_idx = 0
        for l_idx, l_value in enumerate(self.base_l_values):
            irrep_dim = 2 * l_value + 1

            # Extract all channels for this l value
            l_channels = []
            for channel in range(self.multiplicity):
                end_idx = start_idx + irrep_dim
                l_channels.append(node_irreps[:, start_idx:end_idx])
                start_idx = end_idx

            # Combine channels (simple average)
            l_features = torch.mean(
                torch.stack(l_channels, dim=0), dim=0
            )  # [N, irrep_dim]

            if l_value == 0:
                # l=0 features are already invariant
                invariant_features_list.append(l_features)  # [N, 1]
            elif l_value == 1:
                # l=1 features: compute invariant via dot product with self
                l1_invariants = torch.einsum(
                    "ijk,ni,nj->nk", self.cg_l1_l1_l0, l_features, l_features
                )  # [N, 1]
                invariant_features_list.append(l1_invariants)
            # For higher l values, we could compute more invariants, but for now just use l=0,1

        # Combine invariant features from l=0 and l=1 only (first two)
        invariant_features = torch.cat(invariant_features_list[:2], dim=1)  # [N, 2]

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
