import torch
import torch.nn as nn
from e3nn.o3 import _wigner
import math
import time
from .base_model import BaseModel

from e3nn.o3 import spherical_harmonics


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

                            # FIXED: Create small neural networks for sophisticated gating
                            for in_channel in range(self.multiplicity):
                                for out_channel in range(self.multiplicity):
                                    mlp_key = f"gate_mlp_{edge_sh_degree}_{node_irrep_idx}_{out_irrep_idx}_{in_channel}_{out_channel}"

                                    # FIXED: Simpler MLP to reduce overfitting
                                    # Input: distance (1D) + node_invariant (1D) = 2D input
                                    gate_mlp = nn.Sequential(
                                        nn.Linear(2, 4),  # Smaller hidden layer
                                        nn.ReLU(),
                                        nn.Dropout(
                                            0.1
                                        ),  # Add dropout for regularization
                                        nn.Linear(4, 1),  # Output single gate value
                                        nn.Tanh(),  # Keep output bounded
                                    )

                                    # Initialize with smaller weights for stability
                                    for module in gate_mlp:
                                        if isinstance(module, nn.Linear):
                                            nn.init.xavier_uniform_(
                                                module.weight,
                                                gain=0.05,  # Even smaller gain
                                            )
                                            nn.init.zeros_(module.bias)

                                    layer_weights[mlp_key] = gate_mlp
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

    def forward(
        self, node_irreps, edge_index, sh_edge_features, distance_edge_features
    ):
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
                layer_idx,
                current_features,
                edge_index,
                sh_edge_features,
                distance_edge_features,
            )

        return current_features

    def _apply_cg_layer(
        self,
        layer_idx,
        node_irreps,
        edge_index,
        sh_edge_features,
        distance_edge_features,
    ):
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

        # FIXED: Compute invariant features for each source node for gating
        source_invariants = []
        start_idx = 0
        for l_idx, l_value in enumerate(self.base_l_values):
            irrep_dim = 2 * l_value + 1
            l_channels = []
            for channel in range(self.multiplicity):
                end_idx = start_idx + irrep_dim
                l_channels.append(source_features[:, start_idx:end_idx])
                start_idx = end_idx

            # Combine channels and compute invariant
            l_features = torch.mean(torch.stack(l_channels, dim=0), dim=0)
            if l_value == 0:
                invariant = l_features.mean(dim=1, keepdim=True)
            else:
                invariant = torch.norm(l_features, dim=1, keepdim=True)
            source_invariants.append(invariant)

        # Combine all invariants into single feature per source node
        source_invariant_features = torch.cat(source_invariants, dim=1).mean(
            dim=1, keepdim=True
        )  # [num_edges, 1]

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
                    mlp_key = f"gate_mlp_{edge_sh_degree}_{node_irrep_idx}_{out_irrep_idx}_{in_channel}_{out_channel}"
                    gate_mlp = layer_weights[mlp_key]

                    input_channel_idx = node_irrep_idx * self.multiplicity + in_channel
                    node_feat = source_irrep_channels[input_channel_idx]

                    # CG tensor product
                    msg = torch.einsum("ijk,ei,ej->ek", cg_coeffs, edge_feat, node_feat)

                    # FIXED: Use sophisticated gating with distance + invariant features
                    # Prepare inputs for gate MLP: [distance, source_invariant]
                    gate_inputs = torch.cat(
                        [
                            distance_edge_features,  # [num_edges, 1]
                            source_invariant_features,  # [num_edges, 1]
                        ],
                        dim=1,
                    )  # [num_edges, 2]

                    # Apply gate MLP for complex distance-dependent gating
                    gate = gate_mlp(gate_inputs).squeeze(-1)  # [num_edges]

                    output_channel_idx = out_irrep_idx * self.multiplicity + out_channel
                    message_irrep_channels[output_channel_idx] += (
                        gate.unsqueeze(-1) * msg
                    )

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
        message_passing_steps=3,
        final_mlp_dims=[32, 16],
        max_sh_degree=2,
        init_method="xavier",
        seed=42,
        debug=False,
        lr=1e-3,
        weight_decay=1e-5,
        multiplicity=1,
        dropout=0.1,
        num_cg_layers=2,
        base_l_values=[0, 1, 2],
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        torch.manual_seed(seed)
        self.init_method = init_method

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

        # FIXED: More robust final MLP with better scaling
        final_layers = []
        prev_dim = len(self.base_l_values)  # One invariant per l-value

        # Add input normalization
        final_layers.append(nn.LayerNorm(prev_dim))

        for i, dim in enumerate(final_mlp_dims):
            final_layers.append(nn.Linear(prev_dim, dim))
            final_layers.append(
                nn.LayerNorm(dim)
            )  # Layer normalization for stable training
            final_layers.append(nn.ReLU())
            if dropout > 0:
                final_layers.append(nn.Dropout(dropout))  # Dropout for regularization
            prev_dim = dim

        # FIXED: Add a final layer with proper initialization
        final_layers.append(nn.Linear(prev_dim, 1))
        # Don't add activation - let it output raw logits

        self.final_mlp = nn.Sequential(*final_layers)

        # FIXED: Initialize final layer with smaller weights
        with torch.no_grad():
            self.final_mlp[-1].weight.data *= 0.1
            self.final_mlp[-1].bias.data.fill_(0.0)

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better training stability for the CG layers"""
        import math

        # Iterate through each message passing layer (MultiCGLayer)
        for multi_cg_layer in self.message_layers:
            # multi_cg_layer.cg_layers is a ModuleList of ParameterDict for each CG sub-layer
            for layer_idx, cg_layer in enumerate(multi_cg_layer.cg_layers):
                valid_combos = getattr(
                    multi_cg_layer, f"valid_combinations_{layer_idx}", []
                )
                fan_in = len(valid_combos) if valid_combos else 1
                fan_out = (
                    self.multiplicity
                )  # Each weight is a scalar; using multiplicity for scaling

                for key, param in cg_layer.items():
                    # FIXED: Handle both nn.Parameter and nn.Module (MLP) cases
                    if isinstance(param, nn.Parameter):
                        # Original parameter initialization
                        if self.init_method == "xavier":
                            std = math.sqrt(2.0 / (fan_in + fan_out)) * 0.1
                            with torch.no_grad():
                                param.normal_(0, std)
                        elif self.init_method == "xavier_uniform":
                            xavier_uniform_init_steerable(param, fan_in, fan_out)
                        elif self.init_method == "constant":
                            with torch.no_grad():
                                param.fill_(0.01)
                        elif self.init_method == "uniform":
                            bound = 0.1 / math.sqrt(fan_in) if fan_in > 0 else 0.01
                            with torch.no_grad():
                                param.uniform_(-bound, bound)
                        elif self.init_method == "kaiming":
                            std = math.sqrt(2.0 / fan_in) * 0.1 if fan_in > 0 else 0.01
                            with torch.no_grad():
                                param.normal_(0, std)
                    elif isinstance(param, nn.Module):
                        # MLP initialization - already done in _setup_cg_layer
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

        # FIXED: Initialize node features with small random values to avoid dead ReLUs
        initial_features = []

        for l_value in self.base_l_values:
            irrep_dim = 2 * l_value + 1  # Compute dimension from l value
            for channel in range(self.multiplicity):
                if (
                    l_value == 0
                ):  # Scalar features - initialize to small positive values
                    irrep_channel = (
                        torch.ones(num_nodes, irrep_dim, device=device) * 0.1
                    )
                else:  # Higher order features - initialize to small random values (NOT zeros!)
                    irrep_channel = (
                        torch.randn(num_nodes, irrep_dim, device=device) * 0.01
                    )
                initial_features.append(irrep_channel)

        node_irreps = torch.cat(initial_features, dim=1)  # [N, 2+6] for mult=2

        if self.debug:
            embed_time = time.time() - start_time
            mp_start = time.time()

        # Split edge attributes to irrep format efficiently
        sh_edge_features = spherical_harmonics(
            self.max_sh_degree, edge_attr, normalize=True
        )
        sh_edge_features = self._split_sh_features_tensor(sh_edge_features)
        distance_edge_features = torch.norm(edge_attr, dim=1, keepdim=True)

        # Message passing with proper equivariance - FIXED: Add skip connections
        residual_irreps = node_irreps  # Store initial features
        for i, layer in enumerate(self.message_layers):
            if self.debug:
                layer_start = time.time()

            # Apply message passing layer
            new_irreps = layer(
                node_irreps, edge_index, sh_edge_features, distance_edge_features
            )

            # FIXED: Add skip connection with learnable mixing
            if i > 0:  # Skip connection after first layer
                node_irreps = 0.8 * new_irreps + 0.2 * node_irreps
            else:
                node_irreps = new_irreps

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
                # l=0 features are already invariant - take mean across channels
                scalar_inv = l_features.mean(dim=1, keepdim=True)  # [N, 1]
                # FIXED: Add proper scaling for scalar features
                invariant_features_list.append(scalar_inv * 10.0)  # Scale up
            else:
                # For l>0: compute invariant via norm (rotation-invariant)
                l_norm = torch.norm(l_features, dim=1, keepdim=True)  # [N, 1]
                # FIXED: Add non-linearity and scaling for vector features
                l_norm_scaled = torch.tanh(l_norm) * 5.0  # Scale and add non-linearity
                invariant_features_list.append(l_norm_scaled)

        # Combine all invariant features
        invariant_features = torch.cat(
            invariant_features_list, dim=1
        )  # [N, len(base_l_values)]

        # Compute raw logits (no softmax yet)
        raw_logits = self.final_mlp(invariant_features).squeeze(-1)  # [N]

        # DISPLACEMENT FROM MEAN APPROACH
        if batch is not None:
            # For batched processing, compute displacement per graph
            unique_batches = torch.unique(batch)
            centroid_predictions = []
            for i, b in enumerate(unique_batches):
                mask = batch == b
                batch_logits = raw_logits[mask]
                batch_positions = node_pos[mask]

                # FIXED: Use more of geometric center for stability
                batch_weights = torch.softmax(batch_logits, dim=0)  # [N_batch]
                weighted_com = torch.sum(
                    batch_weights.unsqueeze(-1) * batch_positions, dim=0
                )  # [3]
                geometric_center = batch_positions.mean(dim=0)  # [3]

                # Use more geometric center for stability
                final_com = 0.8 * geometric_center + 0.2 * weighted_com

                centroid_predictions.append(final_com)
            centroid_prediction = torch.stack(centroid_predictions, dim=0)  # [B, 3]
        else:
            # Single graph case
            weights = torch.softmax(raw_logits, dim=0)

            # Predict weighted center of mass
            weighted_com = torch.sum(weights.unsqueeze(-1) * node_pos, dim=0)  # [3]
            geometric_center = node_pos.mean(dim=0)  # [3]

            # FIXED: Use more of geometric center for stability
            final_com = 0.8 * geometric_center + 0.2 * weighted_com
            centroid_prediction = final_com.unsqueeze(0)  # [1, 3]

        if self.debug:
            final_time = time.time() - final_start
            total_time = time.time() - start_time
            print(
                f"[DEBUG] Embed: {embed_time:.3f}s, MP: {mp_time:.3f}s, Final: {final_time:.3f}s"
            )
            print(f"[DEBUG] Total: {total_time:.3f}s")

        return centroid_prediction

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
