import pandas as pd
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
import numpy as np

distance_df = pd.read_csv('brick_rp_distances.csv')
index_df = pd.read_csv('bricks_index_values.csv')

distances = distance_df.iloc[:, 1:].values
index_values = index_df['index_value'].values
bricks = distance_df['brick'].values

num_srs = distances.shape[1]  
num_bricks = distances.shape[0]  
workload_intervals = [[0.8, 1.2], [0.85, 1.15], [0.9, 1.1]]  

current_assignments = {
    1: [4, 5, 6, 7, 8, 15],
    2: [10, 11, 12, 13, 14],
    3: [9, 16, 17, 18],
    4: [1, 2, 3, 19, 20, 21, 22]
}


current_assignment_matrix = [[0] * num_bricks for _ in range(num_srs)]
for sr, assigned_bricks in current_assignments.items():
    for brick in assigned_bricks:
        current_assignment_matrix[sr - 1][brick - 1] = 1


def create_model(primary_obj, epsilon=None, workload_limits=[0.8, 1.2]):
    model = Model("Epsilon_Constraint")
    
    # Decision variables
    x = model.addVars(num_srs, num_bricks, vtype=GRB.BINARY, name="x")
    delta = model.addVars(num_srs, num_bricks, vtype=GRB.CONTINUOUS, name="delta")

    # Objectives
    distance_obj = quicksum(distances[j, i] * x[i, j] for i in range(num_srs) for j in range(num_bricks))
    disruption_obj = quicksum(index_values[j] * delta[i, j] for i in range(num_srs) for j in range(num_bricks))
    
    # Set primary objective
    if primary_obj == "distance":
        model.setObjective(distance_obj, GRB.MINIMIZE)
    elif primary_obj == "disruption":
        model.setObjective(disruption_obj, GRB.MINIMIZE)

    # Constraints
    # Each brick is assigned to exactly one SR
    for j in range(num_bricks):
        model.addConstr(quicksum(x[i, j] for i in range(num_srs)) == 1, name=f"Assign_{j}")

    # Workload limits for each SR
    for i in range(num_srs):
        workload = quicksum(index_values[j] * x[i, j] for j in range(num_bricks))
        model.addConstr(workload >= workload_limits[0], name=f"MinWorkload_{i}")
        model.addConstr(workload <= workload_limits[1], name=f"MaxWorkload_{i}")

    # Linearize absolute value |x_ij - x_ij^0|
    for i in range(num_srs):
        for j in range(num_bricks):
            model.addConstr(delta[i, j] >= x[i, j] - current_assignment_matrix[i][j], name=f"DeltaPos_{i}_{j}")
            model.addConstr(delta[i, j] >= current_assignment_matrix[i][j] - x[i, j], name=f"DeltaNeg_{i}_{j}")

    # Epsilon constraint
    if epsilon is not None:
        if primary_obj == "distance":
            model.addConstr(disruption_obj <= epsilon, name="Epsilon_Disruption")
        elif primary_obj == "disruption":
            model.addConstr(distance_obj <= epsilon, name="Epsilon_Distance")

    return model, x

# Epsilon-Constraint Method
def epsilon_constraint_method(epsilon_values, primary_obj, workload_limits):
    pareto_results = []
    for epsilon in epsilon_values:
        model, _ = create_model(primary_obj=primary_obj, epsilon=epsilon, workload_limits=workload_limits)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            distance = model.getObjective().getValue() if primary_obj == "distance" else model.getConstrByName("Epsilon_Distance").rhs
            disruption = model.getObjective().getValue() if primary_obj == "disruption" else model.getConstrByName("Epsilon_Disruption").rhs
            pareto_results.append((distance, disruption))
    return pareto_results


def visualize_pareto_combined(pareto_distance, pareto_disruption, workload_limits):
    plt.figure(figsize=(10, 6))

    if pareto_distance:
        distances, disruptions_distance = zip(*pareto_distance)
        plt.scatter(distances, disruptions_distance, c='blue', label='Minimizing Distance')

    if pareto_disruption:
        distances_disruption, disruptions = zip(*pareto_disruption)
        plt.scatter(distances_disruption, disruptions, c='red', label='Minimizing Disruption')

    plt.title(f"Combined Pareto Front: Distance vs Disruption ({workload_limits[0]}, {workload_limits[1]})")
    plt.xlabel("Total Distance")
    plt.ylabel("Total Disruption")
    plt.legend()
    plt.grid()
    plt.show()

for workload_limits in workload_intervals:
    epsilon_values = np.linspace(0, 10, 100) 

    pareto_distance = epsilon_constraint_method(epsilon_values, primary_obj="distance", workload_limits=workload_limits)

    pareto_disruption = epsilon_constraint_method(epsilon_values, primary_obj="disruption", workload_limits=workload_limits)

    visualize_pareto_combined(pareto_distance, pareto_disruption, workload_limits)
