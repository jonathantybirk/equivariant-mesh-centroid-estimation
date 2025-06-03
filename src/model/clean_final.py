# %%
from e3nn.o3 import _wigner, spherical_harmonics
import torch
import math


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

        # Initialize weights parameter
        self.weight = torch.nn.parameter.Parameter(
            torch.zeros([len(input_a_l) * len(input_h_l)], dtype=torch.double)
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
        
        # Create output tensor
        out_dim = self.l_out * 2 + 1
        result = torch.zeros([out_dim], dtype=torch.double)

        # Process only valid combinations
        for combo_idx, (a_l_idx, h_l_idx) in enumerate(self.valid_combos):
            a_l_in = self.input_a_l[a_l_idx]
            h_l_in = self.input_h_l[h_l_idx]

            # Safety check
            if a_l_idx >= len(input_a) or h_l_idx >= len(input_h):
                continue

            try:
                # Get the CG coefficients
                GC = _wigner._so3_clebsch_gordan(a_l_in, h_l_in, self.l_out)

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


def invariant_feat(x_l_in, x_in):
    GC = _wigner._so3_clebsch_gordan(x_l_in, x_l_in, 0)
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


class MessageFunction(torch.nn.Module):
    def __init__(self, input_a_l, input_h_l, h_l_out, init_method="xavier"):
        super(MessageFunction, self).__init__()
        # Use l-values that can form valid CG coefficients
        # Use same as h_l_out to preserve equivariance
        self.intermediate_dims = h_l_out
        self.hidden_h_layer_1 = HiddenHLayer(
            input_a_l, input_h_l, self.intermediate_dims, init_method
        )
        self.hidden_h_layer_2 = HiddenHLayer(input_a_l, self.intermediate_dims, h_l_out, init_method)
        self.h_l_out = h_l_out

    def forward(self, input_a, input_h):
        # First layer outputs a list of tensors
        h1 = self.hidden_h_layer_1(input_a, input_h)

        # Second layer with h1 list
        h2 = self.hidden_h_layer_2(input_a, h1)

        return h2 