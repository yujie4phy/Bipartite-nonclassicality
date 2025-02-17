import numpy as np
from gurobipy import Model, GRB, LinExpr

num_constraints = 580
num_variables = 400

model = Model("Dual_Variable_Optimization")
model.setParam('OutputFlag', 1)


y = model.addVars(num_constraints, name="y")

# Step 4: Define the objective function (minimize y^T b)
objective = LinExpr()
for i in range(num_constraints):
    objective += y[i] * b[i]
model.setObjective(objective, GRB.MINIMIZE)

# Step 5: Add constraints (0 <= y^T M <=1 for each column j)
for j in range(num_variables):
    yTM = LinExpr()
    for i in range(num_constraints):
        yTM += y[i] * M[i, j]
    model.addConstr(yTM >= 0, name=f"LowerBound_yTM_{j}")
    model.addConstr(yTM <= 1, name=f"UpperBound_yTM_{j}")

# Step 6: Optimize the model
model.optimize()

# Step 7: Retrieve and interpret the results
if model.status == GRB.OPTIMAL:
    y_opt = np.array([y[i].X for i in range(num_constraints)])
    print("\nOptimal dual variable y:")
    print(y_opt)
    print(f"\nOptimal objective value (y^T b): {model.ObjVal}")
else:
    print("\nNo optimal solution found.")
    if model.status == GRB.INFEASIBLE:
        print("The model is infeasible.")
    elif model.status == GRB.UNBOUNDED:
        print("The model is unbounded.")
    else:
        print(f"Optimization ended with status {model.status}")
