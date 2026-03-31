import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stochastic Transportation Model", layout="wide")

st.title("🚛 Stochastic Transportation Problem using Water Cycle Algorithm")

# =============================================================
# INPUT SECTION
# =============================================================
st.sidebar.header("Simulation Controls")

n_simulations = st.sidebar.slider("Monte Carlo Simulations", 1000, 20000, 10000)
iterations = st.sidebar.slider("WCA Iterations", 100, 2000, 1000)

# =============================================================
# PARAMETERS
# =============================================================
supply_params = [
    {"alpha": 550, "beta": 5.8, "gamma": 650, "eta": 0.91},
    {"alpha": 650, "beta": 6.4, "gamma": 750, "eta": 0.92},
    {"alpha": 750, "beta": 7.2, "gamma": 850, "eta": 0.93},
]

demand_params = [
    {"alpha": 325, "beta": 6.2, "gamma": 725, "delta": 0.933},
    {"alpha": 425, "beta": 6.9, "gamma": 825, "delta": 0.902},
    {"alpha": 525, "beta": 7.3, "gamma": 925, "delta": 0.912},
    {"alpha": 625, "beta": 7.8, "gamma": 1025, "delta": 0.909},
]

location_names = ["Bir", "Thakurdwara", "Dharamsala"]
dest_names = ["Delhi", "Punjab", "Rajasthan", "UP"]

# =============================================================
# FUNCTIONS
# =============================================================
def weibull_supply_bound(alpha, beta, gamma, eta):
    return alpha + beta * ((-np.log(eta)) ** (1 / gamma))

def weibull_demand_bound(alpha, beta, gamma, delta):
    return alpha + beta * ((-np.log(1 - delta)) ** (1 / gamma))

# =============================================================
# BLOCK 1: WEIBULL VISUALIZATION
# =============================================================
st.header("Weibull Distribution")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, (params, ax) in enumerate(zip(supply_params, axes)):
    alpha, beta, gamma = params["alpha"], params["beta"], params["gamma"]
    x = np.linspace(alpha, alpha + 3 * beta, 500)
    z = (x - alpha) / beta
    pdf = (gamma / beta) * (z ** (gamma - 1)) * np.exp(-(z ** gamma))

    ax.plot(x, pdf)
    ax.set_title(location_names[i])
    ax.grid(True)

st.pyplot(fig)

# =============================================================
# BLOCK 2: BOUNDS
# =============================================================
st.header("Deterministic Bounds")

supply_bounds = [weibull_supply_bound(**p) for p in supply_params]
demand_bounds = [weibull_demand_bound(**p) for p in demand_params]

st.write("Supply Bounds:", supply_bounds)
st.write("Demand Bounds:", demand_bounds)

# =============================================================
# BLOCK 3: MONTE CARLO
# =============================================================
st.header("Monte Carlo Simulation")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, (params, ax) in enumerate(zip(supply_params, axes)):
    alpha, beta, gamma = params["alpha"], params["beta"], params["gamma"]
    U = np.random.uniform(0, 1, n_simulations)
    samples = alpha + beta * ((-np.log(U)) ** (1 / gamma))

    ax.hist(samples, bins=40)
    ax.axvline(supply_bounds[i], linestyle='--')
    ax.set_title(location_names[i])

st.pyplot(fig)

# =============================================================
# BLOCK 4: WCA OPTIMIZATION
# =============================================================
st.header("Water Cycle Algorithm Optimization")

cost_matrix = np.array([
    [10, 10, 20, 11],
    [12, 7, 9, 20],
    [10, 14, 16, 18]
])

n_supply = 3
n_demand = 4
n_vars = n_supply * n_demand

def fitness(x):
    X = x.reshape(n_supply, n_demand)
    cost = np.sum(cost_matrix * X)
    penalty = 0

    for t in range(n_supply):
        if np.sum(X[t]) > supply_bounds[t]:
            penalty += 1e6

    for s in range(n_demand):
        if np.sum(X[:, s]) < demand_bounds[s]:
            penalty += 1e6

    return cost + penalty

lb, ub = 0, max(supply_bounds)

population = np.random.uniform(lb, ub, (50, n_vars))
fitness_vals = np.array([fitness(ind) for ind in population])

best_history = []

for _ in range(iterations):
    idx = np.argmin(fitness_vals)
    best = population[idx]

    for i in range(len(population)):
        population[i] += np.random.rand(n_vars) * (best - population[i])

    fitness_vals = np.array([fitness(ind) for ind in population])
    best_history.append(np.min(fitness_vals))

# =============================================================
# CONVERGENCE PLOT
# =============================================================
fig = plt.figure()
plt.plot(best_history)
plt.title("Convergence")
plt.xlabel("Iterations")
plt.ylabel("Cost")

st.pyplot(fig)

# =============================================================
# RESULTS
# =============================================================
best_idx = np.argmin(fitness_vals)
optimal = population[best_idx].reshape(n_supply, n_demand)

st.header("Optimal Transportation Plan")
st.dataframe(optimal)

st.success(f"Minimum Cost: ₹ {np.sum(cost_matrix * optimal):.2f}")