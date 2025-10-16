# Graphon-Based Mean Field Epidemic Solver (FBODE)

This repository contains Python code for simulating and solving a forward–backward system of ordinary differential equations (FBODE) for epidemic control over a heterogeneous population. The implementation follows the numerical framework described in *Finite State Graphon Games with Applications to Epidemics* (Aurell, Carmona, Dayanıklı, Laurière, 2022).

## Overview
The project models the spread of an epidemic in a population represented as a continuum of agents interacting through a **graphon**—a continuous limit of large networks that captures heterogeneity in contact patterns.  
Each agent controls its contact rate to minimize infection risk and behavioral deviation costs, while the system evolves according to coupled forward–backward equations:
- The **forward equations** describe the population dynamics (fractions of Susceptible, Infected, and Recovered individuals).
- The **backward equations** describe the value functions driving optimal control behavior.

A neural network provides the initial condition for the backward equation, enabling data-driven initialization and improved convergence.

### Key Features
- Implements the **Forward–Backward ODE system** for SIR-type dynamics.
- Integrates **graphon-based heterogeneity**: constant, power-law, star, or piecewise kernels.
- Uses a **neural network** to approximate the initial value function \( U_0(x) \).
- Supports configurable training hyperparameters and epidemic parameters via `config.yaml`.
- Outputs simulation data (`.npz`) including trajectories, optimal controls, mean-field interactions, and training loss.

## Dependencies
Required Python packages:
- `numpy`
- `tensorflow`
- `pyyaml`
- `matplotlib`
- `os`, `time`, `math`

## Code Structure

- **`main-fbode-solver.py`**  
  Entry point. Loads configuration, builds graphon, initializes parameters, runs training and ODE simulation, and saves outputs.

- **`FBODE.py`**  
  Defines the forward–backward ODE system (`FBODEEquation`).  
  - `driver_P`: Forward evolution of population densities (S, I, R).  
  - `driver_U`: Backward evolution of value functions.  
  - `optimal_ALPHA`: Computes optimal individual control rates.

- **`fbodesolver.py`**  
  Core solver class implementing training and time stepping.  
  - Builds graphon weight matrices and samples agent identities.  
  - Integrates forward and backward ODEs over time.  
  - Uses a neural network (`MyModelU0`) to learn initial conditions.  
  - Tracks and saves loss history and simulation data.

- **`graphon_utils.py`**  
  (Optional utility) Builds graphon functions based on configuration: constant, power-law, star, or piecewise.

- **`config.yaml`**  
  YAML configuration file specifying parameters for solver, epidemic model, neural network, and graphon structure.

- **`data/`**  
  Directory where training and simulation results are stored.  
  Files include:  
  - `data_fbodesolver_solution_final.npz`: Final simulation output.  
  - `data_fbodesolver_solution_iterX.npz`: Intermediate checkpoints.  
  - `res-console.txt`: Console log of loss and runtime.  

## Usage
To run the project:
```python
python main-fbodesolver.py --cfg config.yaml
```

## Output Description
Each `.npz` output file contains:
- `t_path`: Time grid.  
- `P_path`: Population densities (S, I, R).  
- `U_path`: Value function trajectories.  
- `ALPHA_path`: Optimal control values.  
- `Z_path`: Mean-field infection exposure.  
- `loss_history`: Neural network loss evolution.

