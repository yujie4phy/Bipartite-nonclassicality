import numpy as np

# Function to generate the 16 local deterministic points
def generate_local_vertices():
    local_points = []
    for alpha, beta, gamma, delta in np.ndindex(2, 2, 2, 2):  # iterating over {0, 1}^4
        point = np.zeros(16)
        for x in [0, 1]:
            for y in [0, 1]:
                a = (alpha * x) ^ beta  # a = αx ⊕ β
                b = (gamma * y) ^ delta  # b = γy ⊕ δ
                point[8 * x + 4 * y + a * 2 + b] = 1
        local_points.append(point)
    return local_points

# Function to generate the 8 non-local PR box points
def generate_pr_vertices():
    pr_points = []
    for alpha, beta, gamma in np.ndindex(2, 2, 2):  # iterating over {0, 1}^3
        point = np.zeros(16)
        for x in [0, 1]:
            for y in [0, 1]:
                for a in [0, 1]:
                    b = (a ^ (x * y) ^ (alpha * x) ^ (beta * y) ^ gamma)  # a ⊕ b = x.y ⊕ αx ⊕ βy ⊕ γ
                    point[8 * x + 4 * y + a * 2 + b] = 0.5
        pr_points.append(point)
    return pr_points

# Combine the 16 local vertices and 8 PR boxes
local_points = generate_local_vertices()
pr_points = generate_pr_vertices()
all_points = local_points + pr_points

# Function to check the inequality
def check_inequality(p):
    term1 = (1/4) * (p[0] + p[3] + p[8] + p[11] + p[4] + p[7] + p[14] + p[13])
    term2 = (1/2) * (p[0] + p[1])
    return term1  == 3/4

# Check how many points satisfy the inequality
satisfying_points = sum(check_inequality(p) for p in local_points)

# Output the result
print(f"Number of points that satisfy the inequality: {satisfying_points} out of 24")
