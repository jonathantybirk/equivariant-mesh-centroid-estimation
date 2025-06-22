# %%


def validate_triangle_inequality(l1, l2, l3):
    """
    Check triangle inequality for Clebsch-Gordan coefficients.
    For valid coupling: |l1 - l2| ≤ l3 ≤ l1 + l2
    """
    min_l = abs(l1 - l2)
    max_l = l1 + l2
    if not (min_l <= l3 <= max_l):
        return 0
    return 1


l1s = [0, 1, 2, 0, 1, 2, 0]
# l1s = [0, 1, 2]
l2a = [0, 1, 2]
l3s = [0]

# check how many combinations are valid
valid_connections = 0
l1_dim = 0
l2_dim = 0
l3_dim = 0
for l1 in l1s:
    for l2 in l2a:
        for l3 in l3s:
            valid_connections += validate_triangle_inequality(l1, l2, l3)
            if validate_triangle_inequality(l1, l2, l3):
                print(
                    f"l1: {l1}, l2: {l2}, l3: {l3}, valid: {validate_triangle_inequality(l1, l2, l3)}"
                )

l1_dim = sum([2 * l1 + 1 for l1 in l1s])
l2_dim = sum([2 * l2 + 1 for l2 in l2a])
l3_dim = sum([2 * l3 + 1 for l3 in l3s])

print("l3 valid: ", valid_connections)
print("l1_dim: ", l1_dim)
print("l2_dim: ", l2_dim)
print("l3_dim: ", l3_dim)

# %%
from e3nn.o3 import _wigner

_wigner._so3_clebsch_gordan(1, 1, 0)
