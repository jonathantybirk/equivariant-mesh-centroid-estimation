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


class CGWeight(torch.nn.Module):
    """
    Highly optimized Clebsch-Gordan tensor product layer with precomputed coefficients.

    Key optimizations:
    1. Precompute all CG coefficients during initialization
    2. Store only valid combinations with their coefficients
    3. Use efficient batched tensor operations
    4. Minimize device transfers and memory allocations
    """

    def __init__(
        self, input_a_l, input_h_l, l_out, init_method="xavier", multiplicity=1
    ):
        super(CGWeight, self).__init__()
        self.input_a_l = input_a_l
        self.input_h_l = input_h_l
        self.l_out = l_out
        self.multiplicity = multiplicity
        self.out_dim = 2 * l_out + 1

        # Precompute and store all valid CG combinations with coefficients
        self.valid_combos = []
        self.cg_coefficients = []

        for a_l_idx, a_l_in in enumerate(input_a_l):
            for h_l_idx, h_l_in in enumerate(input_h_l):
                try:
                    validate_triangle_inequality(a_l_in, h_l_in, l_out)

                    # Get CG coefficients - let cache handle device properly
                    cg_coeffs = CGCoefficientsCache.get_coefficients(
                        a_l_in, h_l_in, l_out, None  # Let cache use appropriate device
                    )

                    if cg_coeffs is not None and cg_coeffs.abs().sum() > 1e-10:
                        self.valid_combos.append(
                            (
                                a_l_idx,
                                h_l_idx,
                                a_l_in,
                                h_l_in,
                            )  # Removed redundant combo_idx
                        )
                        self.cg_coefficients.append(cg_coeffs)
                except (ValueError, Exception):
                    continue

        if len(self.valid_combos) == 0:
            raise ValueError(
                f"No valid CG combinations possible for input_a_l={input_a_l}, "
                f"input_h_l={input_h_l}, l_out={l_out}. All combinations violate triangle inequality."
            )

        # Initialize weights parameter - support multiplicities (multiple channels)
        self.weight = torch.nn.parameter.Parameter(
            torch.zeros(len(self.valid_combos), multiplicity, dtype=torch.float32)
        )

        # Register CG coefficients as buffers (non-trainable parameters that move with model)
        for i, cg_coeffs in enumerate(self.cg_coefficients):
            self.register_buffer(f"cg_coeffs_{i}", cg_coeffs)

        # Apply initialization to all channels
        fan_in = len(self.valid_combos)
        fan_out = self.out_dim * multiplicity  # Account for multiplicity in fan_out

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
        Ultra-fast vectorized forward pass using precomputed CG coefficients with multiplicity support
        """
        if len(input_a_batch) == 0 or len(input_h_batch) == 0:
            return torch.zeros(
                0,
                self.out_dim * self.multiplicity,
                dtype=torch.float32,
                device=self.weight.device,
            )

        num_edges = input_a_batch[0].shape[0]

        # Result tensor accounts for multiplicity: [num_edges, out_dim * multiplicity]
        result = torch.zeros(
            num_edges,
            self.out_dim * self.multiplicity,
            dtype=torch.float32,
            device=self.weight.device,
        )

        # Process all valid combinations efficiently
        for i, (a_l_idx, h_l_idx, a_l_in, h_l_in) in enumerate(self.valid_combos):
            # Get precomputed CG coefficients (already on correct device)
            cg_coeffs = getattr(self, f"cg_coeffs_{i}")

            # Safety checks with early exit
            if a_l_idx >= len(input_a_batch) or h_l_idx >= len(input_h_batch):
                continue

            # Get input tensors
            a_tensor = input_a_batch[a_l_idx]  # [E, 2*a_l_in+1]
            h_tensor = input_h_batch[h_l_idx]  # [E, 2*h_l_in+1]

            # Apply CG product: einsum is very optimized in PyTorch
            temp = torch.einsum(
                "ijk,ei,ej->ek", cg_coeffs, a_tensor, h_tensor
            )  # [E, out_dim]

            # Weighted accumulation over all channels (multiplicity)
            for channel in range(self.multiplicity):
                start_idx = channel * self.out_dim
                end_idx = (channel + 1) * self.out_dim
                result[:, start_idx:end_idx].add_(temp * self.weight[i, channel])

        return result


class HiddenHLayer(torch.nn.Module):
    def __init__(
        self, input_a_l, input_h_l, h_l_out, init_method="xavier", multiplicity=1
    ):
        super(HiddenHLayer, self).__init__()
        self.multiplicity = multiplicity
        # Use ModuleList instead of regular Python list to properly register parameters
        self.cg_weight = torch.nn.ModuleList(
            [
                CGWeight(input_a_l, input_h_l, l_out, init_method, multiplicity)
                for l_out in h_l_out
            ]
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
    def __init__(
        self, input_a_l, input_h_l, h_l_out, init_method="xavier", multiplicity=1
    ):
        super(MessageFunction, self).__init__()
        # Use l-values that can form valid CG coefficients
        # Use same as h_l_out to preserve equivariance
        self.intermediate_dims = h_l_out
        self.multiplicity = multiplicity
        self.hidden_h_layer_1 = HiddenHLayer(
            input_a_l, input_h_l, self.intermediate_dims, init_method, multiplicity
        )
        self.hidden_h_layer_2 = HiddenHLayer(
            input_a_l,
            self.intermediate_dims,
            self.intermediate_dims,
            init_method,
            multiplicity,
        )
        self.hidden_h_layer_3 = HiddenHLayer(
            input_a_l, self.intermediate_dims, h_l_out, init_method, multiplicity
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

    def __init__(self, max_sh_degree, irrep_dims, hidden_dim, multiplicity=1):
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
        input_dim=3,
        hidden_dim=16,
        message_passing_steps=3,
        final_mlp_dims=[64, 32],
        max_sh_degree=1,
        init_method="xavier",
        seed=42,
        debug=False,
        lr=None,
        weight_decay=None,
        multiplicity=2,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        torch.manual_seed(seed)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_sh_degree = max_sh_degree
        self.debug = debug
        self.message_passing_steps = message_passing_steps
        self.multiplicity = multiplicity

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
                    hidden_dim=hidden_dim,
                    multiplicity=multiplicity,
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
