import torch
import torch.nn as nn
from e3nn import o3
import time
from torch_scatter import scatter_add
from ..base_model import BaseModel


class FastMessageLayer(nn.Module):
    """
    Ultra-fast message passing layer using e3nn's compiled tensor products.

    Replaces all manual CG operations with a single fused CUDA kernel.
    """

    def __init__(self, mul: int = 2, max_l_edge: int = 1):
        super().__init__()

        # ----- irreps -------------------------------------------------------
        # Node features: mul copies of l=0 (scalar) + mul copies of l=1 (vector)
        self.node_irreps = o3.Irreps(f"{mul}x0e + {mul}x1o")

        # Edge features: spherical harmonics up to max_l_edge
        edge_irreps_list = []
        for l in range(max_l_edge + 1):
            p = "e" if l % 2 == 0 else "o"  # parity: even/odd
            edge_irreps_list.append((1, (l, p)))
        self.edge_irreps = o3.Irreps(edge_irreps_list)

        # ----- fused tensor product ----------------------------------------
        # This replaces ALL our manual CG operations with one optimized kernel
        self.tp = o3.FullyConnectedTensorProduct(
            self.node_irreps,  # left operand (h_j)
            self.edge_irreps,  # right operand (SH_ij)
            self.node_irreps,  # output irreps (m_ij)
            internal_weights=True,  # learnable path-mixing weights
        ).compile()  # JIT → custom CUDA kernel for max speed

    def forward(self, h, sh, edge_index):
        """
        Ultra-fast message passing with compiled e3nn kernels.

        Args:
            h: [N, node_irreps.dim] node features
            sh: [E, edge_irreps.dim] pre-computed spherical harmonics
            edge_index: [2, E] source→target edges

        Returns:
            [N, node_irreps.dim] updated node features
        """
        # Get source node features for each edge
        h_source = h[edge_index[0]]  # [E, node_irreps.dim]

        # Compute messages: m_ij = TensorProduct(h_j, Y_ij)
        # This single call replaces all our manual CG loops!
        messages = self.tp(h_source, sh)  # [E, node_irreps.dim]

        # Aggregate messages: h_i ← Σ_j m_ij
        aggregated = scatter_add(messages, edge_index[1], dim=0, dim_size=h.size(0))

        return aggregated


class EquivariantGNNFast(BaseModel):
    """
    Ultra-fast GPU-optimized equivariant GNN using e3nn compiled kernels.

    Key optimizations:
    1. Uses e3nn's FullyConnectedTensorProduct with compiled CUDA kernels
    2. No manual CG coefficient handling
    3. No Python loops over irrep combinations
    4. Automatic gradient optimization and memory management
    """

    def __init__(
        self,
        input_dim=3,
        hidden_dim=16,
        message_passing_steps=3,
        final_mlp_dims=[64, 32],
        max_sh_degree=1,
        init_method="xavier",  # Not used in e3nn version
        seed=42,
        debug=False,
        lr=None,
        weight_decay=None,
        multiplicity=2,
        dropout=0.1,
    ):
        super().__init__(lr=lr, weight_decay=weight_decay)
        torch.manual_seed(seed)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_sh_degree = max_sh_degree
        self.debug = debug
        self.message_passing_steps = message_passing_steps
        self.multiplicity = multiplicity
        self.dropout = dropout

        # ----- irreps setup ------------------------------------------------
        # Node irreps: multiplicity copies of l=0 + multiplicity copies of l=1
        self.node_irreps = o3.Irreps(f"{multiplicity}x0e + {multiplicity}x1o")

        # Edge irreps: spherical harmonics up to max_sh_degree
        edge_irreps_list = []
        for l in range(max_sh_degree + 1):
            p = "e" if l % 2 == 0 else "o"
            edge_irreps_list.append((1, (l, p)))
        self.edge_irreps = o3.Irreps(edge_irreps_list)

        # ----- message passing layers --------------------------------------
        self.message_layers = nn.ModuleList(
            [
                FastMessageLayer(multiplicity, max_sh_degree)
                for _ in range(message_passing_steps)
            ]
        )

        # ----- invariant extraction ----------------------------------------
        # For computing ||l=1 vectors||² using CG: l=1 ⊗ l=1 → l=0
        self.cg_invariant = o3.FullTensorProduct(
            o3.Irreps("1o"),  # l=1 vector
            o3.Irreps("1o"),  # l=1 vector
            o3.Irreps("0e"),  # l=0 scalar (invariant)
        ).compile()  # Compiled for speed

        # ----- final MLP ----------------------------------------------------
        mlp_layers = []
        prev_dim = 2  # l=0 scalar + l=1 invariant
        for i, dim in enumerate(final_mlp_dims):
            mlp_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.LayerNorm(dim),
                    nn.ReLU(),
                ]
            )
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        mlp_layers.append(nn.Linear(prev_dim, 1))
        self.final_mlp = nn.Sequential(*mlp_layers)

    def forward(self, x, edge_index, edge_attr, batch=None, node_pos=None):
        """Ultra-fast forward pass using e3nn compiled kernels"""
        if self.debug:
            start_time = time.time()

        device = x.device
        num_nodes = x.shape[0]

        # Use positions
        if node_pos is None:
            node_pos = x

        # ----- initial node features ----------------------------------------
        # Initialize to the irrep structure expected by e3nn
        # l=0: constant scalars, l=1: zero vectors (will be built up via message passing)
        initial_features = []

        # Add multiplicity copies of l=0 scalars (set to 1)
        for _ in range(self.multiplicity):
            initial_features.append(torch.ones(num_nodes, 1, device=device))

        # Add multiplicity copies of l=1 vectors (set to 0)
        for _ in range(self.multiplicity):
            initial_features.append(torch.zeros(num_nodes, 3, device=device))

        # Combine into single tensor with correct irrep structure
        h = torch.cat(initial_features, dim=1)  # [N, node_irreps.dim]

        if self.debug:
            embed_time = time.time() - start_time
            mp_start = time.time()

        # ----- spherical harmonics ----------------------------------------
        # Split edge attributes into proper irrep format for e3nn
        sh_features = self._split_sh_features_tensor(edge_attr)

        # ----- message passing ---------------------------------------------
        for i, layer in enumerate(self.message_layers):
            if self.debug:
                layer_start = time.time()
            h = layer(h, sh_features, edge_index)
            if self.debug:
                print(f"[DEBUG] Layer {i}: {time.time() - layer_start:.3f}s")

        if self.debug:
            mp_time = time.time() - mp_start
            final_start = time.time()

        # ----- extract invariants ------------------------------------------
        # Extract l=0 scalars (average across multiplicity)
        scalar_start = 0
        scalar_features = []
        for i in range(self.multiplicity):
            scalar_features.append(h[:, scalar_start + i : scalar_start + i + 1])
        l0_features = torch.mean(torch.cat(scalar_features, dim=1), dim=1, keepdim=True)

        # Extract l=1 vectors (average across multiplicity)
        vector_start = self.multiplicity  # After all l=0 features
        vector_features = []
        for i in range(self.multiplicity):
            start_idx = vector_start + i * 3
            end_idx = start_idx + 3
            vector_features.append(h[:, start_idx:end_idx])
        l1_features = torch.mean(torch.stack(vector_features, dim=0), dim=0)

        # Compute l=1 invariant: ||vector||² using compiled CG
        l1_invariants = self.cg_invariant(l1_features, l1_features)  # [N, 1]

        # Combine all invariant features
        invariant_features = torch.cat([l0_features, l1_invariants], dim=1)  # [N, 2]

        # ----- final prediction --------------------------------------------
        raw_logits = self.final_mlp(invariant_features).squeeze(-1)  # [N]

        # Center of mass prediction (same logic as before)
        if batch is not None:
            # Batched processing
            unique_batches = torch.unique(batch)
            com_predictions = []
            for b in unique_batches:
                mask = batch == b
                batch_logits = raw_logits[mask]
                batch_positions = node_pos[mask]

                geometric_mean = batch_positions.mean(dim=0)
                batch_weights = torch.softmax(batch_logits, dim=0)
                displacement = (
                    torch.sum(batch_weights.unsqueeze(-1) * batch_positions, dim=0)
                    - geometric_mean
                )
                batch_com = geometric_mean + displacement
                com_predictions.append(batch_com)
            com_prediction = torch.stack(com_predictions, dim=0)
        else:
            # Single graph
            geometric_mean = node_pos.mean(dim=0)
            weights = torch.softmax(raw_logits, dim=0)
            displacement = (
                torch.sum(weights.unsqueeze(-1) * node_pos, dim=0) - geometric_mean
            )
            com_prediction = (geometric_mean + displacement).unsqueeze(0)

        if self.debug:
            final_time = time.time() - final_start
            total_time = time.time() - start_time
            print(
                f"[DEBUG] Embed: {embed_time:.3f}s, MP: {mp_time:.3f}s, Final: {final_time:.3f}s"
            )
            print(f"[DEBUG] Total: {total_time:.3f}s")

        return com_prediction

    def _split_sh_features_tensor(self, edge_attr):
        """Split spherical harmonics into e3nn irrep format"""
        # Convert to single tensor in the format expected by e3nn
        # e3nn expects [E, edge_irreps.dim] where features are concatenated by l
        return edge_attr  # Assuming edge_attr is already in correct format

    def _initialize_weights(self):
        """e3nn handles weight initialization automatically"""
        pass

    def _precompute_cg_coefficients(self):
        """e3nn handles CG coefficients automatically"""
        pass
