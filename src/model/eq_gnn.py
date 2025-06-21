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


class EquivariantCGLayer(nn.Module):
    """
    Equivariant CG layer following math.md specification exactly

    Implements: m_ij = σ_g(W_a^(2) σ_g(W_a^(1) h_ij))
    And: f_i' = ψ_f(f_i, Σ_j m_ij, d_i)
    """

    def __init__(
        self,
        edge_sh_degree,
        node_l_values,
        node_multiplicity,
        message_mlp_dims=[64, 32],
        dropout=0.1,
        debug=False,
    ):
        super().__init__()
        self.edge_sh_degree = edge_sh_degree
        self.node_l_values = node_l_values  # [0, 1, 2, ...]
        self.node_multiplicity = node_multiplicity
        self.debug = debug

        # Precompute CG tensor and learnable weights
        self._precompute_cg_tensor()

        # Learnable weights W_a^(1) and W_a^(2) for each edge attribute degree
        # Following math.md: m_ij = σ_g(W_a^(2) σ_g(W_a^(1) h_ij))
        self.edge_weights_1 = nn.ParameterDict()
        self.edge_weights_2 = nn.ParameterDict()

        for a_l in range(self.edge_sh_degree + 1):
            # Weights for each edge attribute degree
            self.edge_weights_1[str(a_l)] = nn.Parameter(torch.randn(1) * 0.1)
            self.edge_weights_2[str(a_l)] = nn.Parameter(torch.randn(1) * 0.1)

        # ψ_f MLP for node updates: f_i' = ψ_f(f_i, Σ_j m_ij, d_i)
        # Input: invariant features from f_i + invariant features from messages + distance
        invariant_dim = len(self.node_l_values)  # One invariant per l-value
        input_dim = (
            invariant_dim * 2 + 1
        )  # f_i invariants + message invariants + distance

        # Build configurable message MLP
        message_layers = []
        prev_dim = input_dim

        for dim in message_mlp_dims:
            message_layers.append(nn.Linear(prev_dim, dim))
            message_layers.append(nn.ReLU())
            if dropout > 0:
                message_layers.append(nn.Dropout(dropout))
            prev_dim = dim

        message_layers.append(
            nn.Linear(prev_dim, invariant_dim)
        )  # Output gate for each l-type
        message_layers.append(nn.Sigmoid())

        self.psi_f = nn.Sequential(*message_layers)

    def _precompute_cg_tensor(self):
        """
        Precompute CG tensor for h ⊗ a → output
        where h = [f_i, f_j, d] has l-values from node_l_values + node_l_values + [0]
        """
        # Build h_l_values: [f_i: node_l_values] + [f_j: node_l_values] + [d: 0]
        h_l_values = self.node_l_values + self.node_l_values + [0]

        valid_connections = []
        cg_tensors = []

        if self.debug:
            print(f"[DEBUG] Building CG tensor for h ⊗ a → output")
            print(f"[DEBUG] h_l_values: {h_l_values}")
            print(f"[DEBUG] a_l_values: {list(range(self.edge_sh_degree + 1))}")
            print(f"[DEBUG] output_l_values: {self.node_l_values}")

        # Find all valid CG connections
        for h_idx, h_l in enumerate(h_l_values):
            for a_l in range(self.edge_sh_degree + 1):
                for out_idx, out_l in enumerate(self.node_l_values):
                    try:
                        validate_triangle_inequality(h_l, a_l, out_l)

                        # Get CG coefficients
                        cg = _wigner._so3_clebsch_gordan(h_l, a_l, out_l).float()

                        if cg.abs().sum() > 1e-10:
                            valid_connections.append((h_idx, a_l, out_idx))
                            cg_tensors.append(cg)

                            if self.debug:
                                print(
                                    f"[DEBUG] Valid: h[{h_idx}](l={h_l}) ⊗ a(l={a_l}) → out[{out_idx}](l={out_l})"
                                )

                    except Exception:
                        continue

        if self.debug:
            print(f"[DEBUG] Total valid connections: {len(valid_connections)}")

        # Store as buffers
        self.register_buffer(
            "valid_connections", torch.tensor(valid_connections, dtype=torch.long)
        )

        # Store CG tensors
        for i, cg in enumerate(cg_tensors):
            self.register_buffer(f"cg_{i}", cg)

        self.num_connections = len(valid_connections)

    def forward(self, edge_index, f, d, a):
        """
        Forward pass implementing math.md specification:
        1. Compute messages m_ij = σ_g(W_a^(2) σ_g(W_a^(1) h_ij))
        2. Aggregate messages
        3. Update nodes f_i' = ψ_f(f_i, Σ_j m_ij, d_i)
        """
        device = f.device
        num_nodes = f.shape[0]
        num_edges = edge_index.shape[1]

        source_idx = edge_index[0]
        target_idx = edge_index[1]

        # Step 1: Build h = [f_i, f_j, d] for each edge
        f_source = f[source_idx]  # [num_edges, f_dim]
        f_target = f[target_idx]  # [num_edges, f_dim]
        h = torch.cat([f_source, f_target, d], dim=1)  # [num_edges, h_dim]

        # Step 2: Compute messages via CG products with learnable weights
        out_dim = f.shape[1]
        messages = torch.zeros(num_edges, out_dim, device=device)

        # Dimensions for indexing
        h_irrep_dims = []
        for l in self.node_l_values + self.node_l_values + [0]:
            h_irrep_dims.append(2 * l + 1)

        a_irrep_dims = []
        for l in range(self.edge_sh_degree + 1):
            a_irrep_dims.append(2 * l + 1)

        out_irrep_dims = []
        for l in self.node_l_values:
            out_irrep_dims.append(2 * l + 1)

        # Process each valid connection with learnable weights
        for conn_idx in range(self.num_connections):
            h_idx, a_l, out_idx = self.valid_connections[conn_idx]
            cg = getattr(self, f"cg_{conn_idx}")

            # Get learnable edge weights (convert tensor element to int)
            a_l_int = int(a_l.item()) if hasattr(a_l, "item") else int(a_l)
            w1 = self.edge_weights_1[str(a_l_int)]
            w2 = self.edge_weights_2[str(a_l_int)]

            # Get h irrep features (convert tensor element to int)
            h_idx_int = int(h_idx.item()) if hasattr(h_idx, "item") else int(h_idx)
            h_start = sum(h_irrep_dims[:h_idx_int]) * self.node_multiplicity
            h_end = h_start + h_irrep_dims[h_idx_int] * self.node_multiplicity
            h_features = h[:, h_start:h_end]

            # Get a irrep features
            a_start = sum(a_irrep_dims[:a_l_int])
            a_end = a_start + a_irrep_dims[a_l_int]
            a_features = a[:, a_start:a_end]

            # Apply CG product for each node_multiplicity channel
            for channel in range(self.node_multiplicity):
                h_ch_start = channel * h_irrep_dims[h_idx_int]
                h_ch_end = (channel + 1) * h_irrep_dims[h_idx_int]
                h_ch = h_features[:, h_ch_start:h_ch_end]

                # CG product: h ⊗ a → message
                raw_msg = torch.einsum("hao,eh,ea->eo", cg, h_ch, a_features)

                # Apply two-layer edge gating: m = σ_g(W_a^(2) σ_g(W_a^(1) h))
                # First layer
                gated_msg_1 = torch.tanh(w1 * raw_msg)
                # Second layer
                gated_msg_2 = torch.tanh(w2 * gated_msg_1)

                # Add to output messages
                out_idx_int = (
                    int(out_idx.item()) if hasattr(out_idx, "item") else int(out_idx)
                )
                out_start = (
                    sum(out_irrep_dims[:out_idx_int]) * self.node_multiplicity
                    + channel * out_irrep_dims[out_idx_int]
                )
                out_end = out_start + out_irrep_dims[out_idx_int]
                messages[:, out_start:out_end] += gated_msg_2

        # Step 3: Aggregate messages to nodes
        aggregated_messages = torch.zeros(num_nodes, out_dim, device=device)
        aggregated_messages.index_add_(0, target_idx, messages)

        # Step 4: Node update f_i' = ψ_f(f_i, Σ_j m_ij, d_i)
        # Extract invariant features from current node features
        f_invariants = self._extract_invariant_features(f)

        # Extract invariant features from aggregated messages
        msg_invariants = self._extract_invariant_features(aggregated_messages)

        # Average distance to neighbors for each node
        node_distances = torch.zeros(num_nodes, 1, device=device)
        node_distances.index_add_(0, target_idx, d)
        neighbor_counts = torch.zeros(num_nodes, device=device)
        neighbor_counts.index_add_(0, target_idx, torch.ones(num_edges, device=device))
        avg_distances = node_distances / (neighbor_counts.unsqueeze(1) + 1e-8)

        # ψ_f MLP: takes invariants from f_i, messages, and distance
        psi_input = torch.cat([f_invariants, msg_invariants, avg_distances], dim=1)
        gates = self.psi_f(psi_input)  # [num_nodes, len(node_l_values)]

        # Apply gates to each l-type separately
        updated_f = torch.zeros_like(f)
        start_idx = 0

        for l_idx, l_value in enumerate(self.node_l_values):
            irrep_dim = 2 * l_value + 1
            gate = gates[:, l_idx : l_idx + 1]  # [num_nodes, 1]

            for channel in range(self.node_multiplicity):
                end_idx = start_idx + irrep_dim

                # Gated update: f_new = f_old + gate * message
                updated_f[:, start_idx:end_idx] = (
                    f[:, start_idx:end_idx]
                    + gate * aggregated_messages[:, start_idx:end_idx]
                )
                start_idx = end_idx

        return updated_f

    def _extract_invariant_features(self, f):
        """
        Extract invariant features following math.md specification:
        s_i = [⟨f_{i,l=0}⟩, ||f_{i,l=1}||_2, ..., ||f_{i,l=L}||_2]
        """
        invariants = []
        start_idx = 0

        for l_value in self.node_l_values:
            irrep_dim = 2 * l_value + 1

            # Collect all channels for this l-value
            l_features = []
            for channel in range(self.node_multiplicity):
                end_idx = start_idx + irrep_dim
                l_features.append(f[:, start_idx:end_idx])
                start_idx = end_idx

            # Stack and compute invariant
            l_tensor = torch.stack(
                l_features, dim=1
            )  # [num_nodes, node_multiplicity, irrep_dim]

            if l_value == 0:
                # For l=0 (scalars): take mean ⟨f_{i,l=0}⟩
                invariant = l_tensor.mean(dim=1).mean(
                    dim=1, keepdim=True
                )  # [num_nodes, 1]
            else:
                # For l>0 (vectors/tensors): take L2 norm ||f_{i,l}||_2
                invariant = torch.norm(l_tensor, dim=2).mean(
                    dim=1, keepdim=True
                )  # [num_nodes, 1]

            invariants.append(invariant)

        return torch.cat(invariants, dim=1)  # [num_nodes, len(node_l_values)]


class EquivariantGNN(BaseModel):
    """
    Equivariant GNN following math.md specification exactly
    """

    def __init__(
        self,
        message_passing_steps=2,
        final_mlp_dims=[64, 32],
        message_mlp_dims=[64, 32],
        edge_sh_degree=2,
        init_method="xavier",
        seed=42,
        debug=False,
        lr=1e-3,
        weight_decay=1e-5,
        node_multiplicity=3,
        dropout=0.1,
        node_l_values=[0, 1, 2],
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        torch.manual_seed(seed)

        self.edge_sh_degree = edge_sh_degree
        self.debug = debug
        self.message_passing_steps = message_passing_steps
        self.node_multiplicity = node_multiplicity
        self.dropout = dropout
        self.node_l_values = node_l_values
        self.message_mlp_dims = message_mlp_dims

        # Message passing layers
        self.message_layers = nn.ModuleList(
            [
                EquivariantCGLayer(
                    edge_sh_degree=self.edge_sh_degree,
                    node_l_values=self.node_l_values,
                    node_multiplicity=self.node_multiplicity,
                    message_mlp_dims=self.message_mlp_dims,
                    dropout=self.dropout,
                    debug=self.debug,
                )
                for _ in range(self.message_passing_steps)
            ]
        )

        # Final MLP for centroid prediction (invariant readout)
        # Input: invariant features s_i = [⟨f_{i,l=0}⟩, ||f_{i,l=1}||_2, ..., ||f_{i,l=L}||_2]
        invariant_dim = len(self.node_l_values)

        final_layers = []
        final_layers.append(nn.LayerNorm(invariant_dim))

        prev_dim = invariant_dim
        for dim in final_mlp_dims:
            final_layers.append(nn.Linear(prev_dim, dim))
            final_layers.append(nn.LayerNorm(dim))
            final_layers.append(nn.ReLU())
            if dropout > 0:
                final_layers.append(nn.Dropout(dropout))
            prev_dim = dim

        final_layers.append(nn.Linear(prev_dim, 1))
        self.final_mlp = nn.Sequential(*final_layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, edge_index, r, batch=None):
        """
        Forward pass following math.md specification:
        1. Initialize features with spherical harmonics
        2. Message passing
        3. Invariant readout s_i = [⟨f_{i,l=0}⟩, ||f_{i,l=1}||_2, ...]
        4. Centroid estimation ĉ = Σ_i ω_i x_i
        """
        if self.debug:
            start_time = time.time()

        device = x.device
        num_nodes = x.shape[0]

        # Step 1: Initialize node features with spherical harmonics of positions
        # Following math.md: "For the first layer, we do not have any node feature f,
        # so we simply set them to spherical harmonics node positions"
        f = self._initialize_node_features(x, device)

        # Prepare edge attributes
        a = spherical_harmonics(list(range(self.edge_sh_degree + 1)), r, normalize=True)
        d = torch.norm(r, dim=1, keepdim=True)

        if self.debug:
            print(f"[DEBUG] Initial f shape: {f.shape}")
            print(f"[DEBUG] Edge attr a shape: {a.shape}")
            print(f"[DEBUG] Edge distance d shape: {d.shape}")

        # Step 2: Message passing
        for i, layer in enumerate(self.message_layers):
            if self.debug:
                layer_start = time.time()

            f = layer(edge_index, f, d, a)

            if self.debug:
                print(f"[DEBUG] Layer {i} output shape: {f.shape}")
                print(f"[DEBUG] Layer {i} time: {time.time() - layer_start:.3f}s")

        # Step 3: Invariant readout s_i = [⟨f_{i,l=0}⟩, ||f_{i,l=1}||_2, ...]
        invariant_features = self.message_layers[0]._extract_invariant_features(f)

        if self.debug:
            print(f"[DEBUG] Final invariant features shape: {invariant_features.shape}")

        # Step 4: Compute logits z_i = g_θ(s_i)
        raw_logits = self.final_mlp(invariant_features).squeeze(-1)  # [N]

        # Step 5: Centroid estimation ĉ = Σ_i ω_i x_i where ω_i = softmax(z_i)
        if batch is not None:
            # Batched processing
            unique_batches = torch.unique(batch)
            centroid_predictions = []

            for b in unique_batches:
                mask = batch == b
                batch_logits = raw_logits[mask]
                batch_positions = x[mask]

                # Softmax weights ω_i = softmax(z_i)
                batch_weights = torch.softmax(batch_logits, dim=0)

                # Weighted centroid ĉ = Σ_i ω_i x_i
                weighted_centroid = torch.sum(
                    batch_weights.unsqueeze(-1) * batch_positions, dim=0
                )
                centroid_predictions.append(weighted_centroid)

            centroid_prediction = torch.stack(centroid_predictions, dim=0)
        else:
            # Single graph case
            weights = torch.softmax(raw_logits, dim=0)
            centroid_prediction = torch.sum(weights.unsqueeze(-1) * x, dim=0).unsqueeze(
                0
            )

        if self.debug:
            total_time = time.time() - start_time
            print(f"[DEBUG] Total forward time: {total_time:.3f}s")

        return centroid_prediction

    def _initialize_node_features(self, x, device):
        """
        Initialize node features following math.md specification:
        "For the first layer, we do not have any node feature f,
        so we simply set them to spherical harmonics node positions"
        """
        # Compute spherical harmonics for each l-value with node_multiplicity
        sh_features = spherical_harmonics(self.node_l_values, x, normalize=True)

        # Expand to node_multiplicity channels
        expanded_features = []
        start_idx = 0

        for l_value in self.node_l_values:
            irrep_dim = 2 * l_value + 1
            end_idx = start_idx + irrep_dim

            # Get the l-th irrep
            l_features = sh_features[:, start_idx:end_idx]  # [num_nodes, irrep_dim]

            # Replicate for node_multiplicity channels
            for channel in range(self.node_multiplicity):
                expanded_features.append(l_features)

            start_idx = end_idx

        # Concatenate all features
        f = torch.cat(expanded_features, dim=1)  # [num_nodes, total_feature_dim]

        return f
