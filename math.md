# Equivariant GNN Mathematical Formulation

## Graph-wise notation

Let $G=(V,E)$ be a directed graph with $|V|=N$ nodes and Cartesian coordinates $x_i \in \mathbb{R}^3$.

For every integer $0 \leq l \leq L$ let $V_l = \mathbb{R}^{2l+1}$ be the _type-$l$ steerable space_, carrying the Wigner-$D^{(l)}$ representation of $SO(3)$.

Node features live in the direct sum:

$$
\tilde{f}_i^{(t)} \in V = \bigoplus_{l=0}^L V_l^{\oplus M}
$$

where $M$ is the channel multiplicity.

For an edge $j \to i$ define:

- The relative displacement $r_{ij} = x_j - x_i$
- Its length $d_{ij} = \|r_{ij}\|$
- The _spherical-harmonic edge attribute_:

$$
\tilde{a}_{ij} = \bigoplus_{l=0}^L \left(Y_m^{(l)}(r_{ij}/d_{ij})\right)_{m=-l}^l
$$

which is itself a steerable vector of maximal degree $L$.

## Simple steerable Edge messages

We will just be using a single weight for each of the irreps outputs of the CG product as shown above.

$$
\tilde{m}_{ij}^{(t)} = \sigma_g\left(W_{\tilde{a}_{ij}}^{(2)} \sigma_g\left(W_{\tilde{a}_{ij}}^{(1)} \tilde{h}_{ij}\right)\right)
$$

We will here let the message be the output of the CG product. This means that the message is a long vector of irreps.

For layer $t$, the CG product is between $\tilde{h}_{ij} := \tilde{f}_i^{(t)} \oplus \tilde{f}_j^{(t)} \oplus d_{ij}$ and $\tilde{a}_{ij}$. With $l$ values for $f_i$, $f_j$ as [0,1,2] and $d$ as 1, and $a$ as [0,1,2], $h$ has $l$ as [0,1,2,0,1,2,0]. All connections are valid, resulting in 33 valid connections, matching the CG product's output channels.

We precompute CG coefficients for all valid connections, and create a big CG tensor. This can be reused for all layers.

this results in a tensor l1_dim: 19, l2_dim: 9, l3_dim: 9

We will simply take the GC product of each l with it self, to get an invariant feature. Since our features are l = [0,1,2], we will have 3 invariant features. So this also means we will have to precompute the CG weights for [0,1,2] \* [0,1,2] > [0,0,0].

For the first layer, we do not have any node feature f, so we simply set them to spherical harmonics node positions.

For the second layer, we will use the node features of the first layer as input.

## Message aggregation

Now we will use a simple MLP to help use create the f again. We take as input the node features, the aggregated messages and the orientation attribute.

$$
\tilde{f}_i' = \psi_f \left( \tilde{f}_i, \sum_{j \in \mathcal{N}(i)} \tilde{m}_{ij}, \tilde{d}_i \right).
$$

Where $\phi_f$ is a simple MLP that takes our invariant features and the aggregated messages as input.

$$
\psi_f = \tilde{f}_i \cdot MLP(\sum_{j \in \mathcal{N}(i)} m_i, d_i)
$$

## Invariant read-out and centroid regression

For each node we extract invariant scalars:

$$
s_i = \left[\langle\tilde{f}_{i,l=0}^{(T)}\rangle, \|\tilde{f}_{i,l=1}^{(T)}\|_2, \ldots, \|\tilde{f}_{i,l=L}^{(T)}\|_2\right] \in \mathbb{R}^{L+1}
$$

normalize, then feed a classical MLP $g_\theta$ to obtain logits $z_i = g_\theta(s_i)$.

Softmax weights $\omega_i = \text{softmax}(z_i)$ generate the centroid estimate:

$$
\hat{c} = \sum_i \omega_i x_i
$$

where $x_i$ is the position of the node.

## Equivariance guarantee

Every linear map is a CG product conditioned on steerable attributes;
every non-linearity is either:

1. element-wise on $l=0$ channels or
2. gated by $l=0$ scalars.

Because each operation commutes with the $O(3)$ representation, the whole architecture satisfies:

$$
\tilde{f}^{(0)} \mapsto D(g)\tilde{f}^{(0)} \Longrightarrow \hat{c} \mapsto R\hat{c}, \quad \forall g=(\mathbf{t},R) \in SE(3)
$$

modulo the trivial translation handled by relative coordinates.

## Discussion

The model combines exact $SE(3)$ symmetry with a lightweight distance-aware gating MLP, avoiding costly self-attention and high-order tensors beyond $l=2$. While theoretically sound, the quadratic channel expansion imposed by multiple CG layers still inflates memory for large $L$; empirically we observe diminishing returns beyond $l=2$, in line with known results. Nonetheless, within the chosen bandwidth the architecture retains strict equivariance and competitive runtime.
