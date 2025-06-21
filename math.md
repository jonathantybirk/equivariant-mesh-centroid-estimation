# Equivariant GNN Mathematical Formulation

## Graph-wise notation

Let $G=(V,E)$ be a directed graph with $|V|=N$ nodes and Cartesian coordinates $x_i \in \mathbb{R}^3$.

For every integer $0 \leq l \leq L$ let $V_l = \mathbb{R}^{2l+1}$ be the _type-$l$ steerable space_, carrying the Wigner-$D^{(l)}$ representation of $SO(3)$.

Node features live in the direct sum:

$$
\tilde{f}_i^{(t)} \in V = \bigoplus_{l=0}^L V_l^{\oplus M}
$$

where $M$ is the `node_multiplicity` parameter.

For an edge $j \to i$ define:

- The relative displacement $r_{ij} = x_j - x_i$
- Its length $d_{ij} = \|r_{ij}\|$
- The _spherical-harmonic edge attribute_:

$$
\tilde{a}_{ij} = \bigoplus_{l=0}^{L_{edge}} \left(Y_m^{(l)}(r_{ij}/d_{ij})\right)_{m=-l}^l
$$

which is itself a steerable vector of maximal degree $L_{edge}$ (`edge_sh_degree`).

## Edge Message Computation

For each edge $j \to i$, we compute messages using Clebsch-Gordan products followed by two-layer gating:

**Step 1: CG Product**

$$
\tilde{m}_{ij}^{raw} = \tilde{h}_{ij} \otimes \tilde{a}_{ij}
$$

**Step 2: Two-layer Gating**

$$
\tilde{m}_{ij}^{(t)} = \sigma_g\left(W_{\tilde{a}_{ij}}^{(2)} \sigma_g\left(W_{\tilde{a}_{ij}}^{(1)} \tilde{m}_{ij}^{raw}\right)\right)
$$

where:

- $\tilde{h}_{ij} := \tilde{f}_i^{(t)} \oplus \tilde{f}_j^{(t)} \oplus d_{ij}$ combines source node features, target node features, and edge distance
- $W_{\tilde{a}_{ij}}^{(1)}$ and $W_{\tilde{a}_{ij}}^{(2)}$ are learnable scalar weights for each spherical harmonic degree in $\tilde{a}_{ij}$
- $\sigma_g = \tanh$ is the gating function
- The CG product $\tilde{h}_{ij} \otimes \tilde{a}_{ij} \rightarrow \tilde{m}_{ij}^{raw}$ creates the raw steerable messages

**Detailed Process:**

For layer $t$, with node l-values `node_l_values = [0,1,2]` and edge SH degree `edge_sh_degree = 2`:

1. **Build h**: $h$ has l-values [0,1,2,0,1,2,0] (source features + target features + distance)
2. **Edge attributes**: $a$ has l-values [0,1,2] (edge spherical harmonics)
3. **CG Product**: Valid CG couplings $h \otimes a \rightarrow m^{raw}$ produce raw message channels
4. **Per-degree Gating**: For each edge attribute degree $l \in \{0,1,2\}$, apply weights $W_l^{(1)}, W_l^{(2)}$ to corresponding message components

We precompute CG coefficients for all valid connections and store them as buffers for efficiency.

## Node Feature Initialization

For the first layer, we initialize node features using spherical harmonics of node positions:

$$
\tilde{f}_i^{(0)} = \text{SH}(x_i) \text{ replicated over multiplicity channels}
$$

For subsequent layers, we use the updated features from the previous layer.

## Invariant Feature Extraction

Throughout the model, we use a **unified invariant extraction function** `extract_invariants()` that converts steerable features to rotation-invariant scalars. This same function is applied to:

1. Current node features during message aggregation
2. Aggregated messages during node updates
3. Final node features for centroid prediction

For any steerable feature tensor $\tilde{f}$ with l-types from `node_l_values`, we extract invariant scalars:

$$
\text{extract\_invariants}(\tilde{f}) = [s^{(0)}, s^{(1)}, s^{(2)}, \ldots]
$$

where each component is:

$$
s^{(l)} = \begin{cases}
\langle \tilde{f}_{l=0} \rangle & \text{if } l = 0 \text{ (mean of scalars)} \\
\|\tilde{f}_{l}\|_2 & \text{if } l > 0 \text{ (L2 norm of vectors/tensors)}
\end{cases}
$$

For example, with `node_l_values = [0,1,2]`, this produces $s \in \mathbb{R}^{3}$ invariant features.

**Key properties:**

- **Rotation invariance**: Norms and means are preserved under rotations
- **Efficient computation**: No need for explicit CG self-products
- **Stable gradients**: Well-conditioned for backpropagation
- **Universal applicability**: Same function works for node features, messages, and any steerable tensor

## Message Aggregation and Node Updates

We aggregate messages to each node and update node features using:

$$
\tilde{f}_i^{(t+1)} = \tilde{f}_i^{(t)} + \text{gate}_i \odot \sum_{j \in \mathcal{N}(i)} \tilde{m}_{ij}^{(t)}
$$

where the gate is computed by an MLP $\psi_f$:

$$
\text{gate}_i = \psi_f\left(s_i^{(f)}, s_i^{(m)}, \overline{r}_i\right)
$$

Here:

- $s_i^{(f)} = \text{extract\_invariants}(\tilde{f}_i^{(t)})$ are invariant features from current node features
- $s_i^{(m)} = \text{extract\_invariants}(\sum_{j \in \mathcal{N}(i)} \tilde{m}_{ij}^{(t)})$ are invariant features from aggregated messages
- $\overline{r}_i = \frac{\sum_{j \in \mathcal{N}(i)} \|r_{ij}\|}{|\mathcal{N}(i)|}$ is the average distance to neighbors

The MLP $\psi_f$ has configurable architecture specified by `message_mlp_dims` and outputs one gate value per l-type.

## Centroid Regression

After $T$ message passing steps, we apply the same invariant extraction function to the final node features:

$$
s_i^{(final)} = \text{extract\_invariants}(\tilde{f}_i^{(T)})
$$

We feed these invariant features to a final MLP $g_\theta$ with architecture specified by `final_mlp_dims`:

$$
z_i = g_\theta(s_i^{(final)})
$$

Softmax weights generate the centroid estimate:

$$
\omega_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}, \quad \hat{c} = \sum_i \omega_i x_i
$$

## Implementation Parameters

The model architecture is controlled by:

- `edge_sh_degree`: Angular resolution for edge directions (default: 2)
- `node_l_values`: Types of geometric features nodes can learn (default: [0,1,2])
- `node_multiplicity`: Number of channels per l-type (default: 3)
- `message_mlp_dims`: Architecture of message aggregation MLP (default: [64,32])
- `final_mlp_dims`: Architecture of final prediction MLP (default: [64,32])
- `message_passing_steps`: Number of message passing layers (default: 2)

## Equivariance Guarantee

Every linear operation is a CG product conditioned on steerable edge attributes.
Every non-linearity operates on:

1. Invariant scalars (MLP inputs), or
2. Element-wise on $l=0$ channels (scalar gating)

Because each operation commutes with the $O(3)$ representation, the architecture satisfies:

$$
x_i \mapsto Rx_i + t \Longrightarrow \hat{c} \mapsto R\hat{c} + t, \quad \forall (R,t) \in SE(3)
$$

Translation equivariance is achieved through:

- Relative coordinates $r_{ij} = x_j - x_i$
- Centered input point clouds during preprocessing
- Direct use of node positions $x_i$ in final centroid computation
