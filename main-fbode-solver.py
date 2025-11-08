# main-fbode-solve.py
import FBODE as FBODE
import fbodesolver as fbodesolver
import tensorflow as tf
import numpy as np
import yaml
import os
from graphon_utils import build_graphon

print('\n\ntf.VERSION = ', tf.__version__, '\n\n')
print('\n\ntf.keras.__version__ = ', tf.keras.__version__, '\n\n')

n_seed = 7

def main():
    # --- load config.yaml ---
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # initial conditions and params
    batch_size = cfg["batch_size"]
    T = cfg["T"]
    m0 = cfg["m0"]
    beta = cfg["beta"]
    gamma = cfg["gamma"]
    g = cfg["g"]
    Delta_t = cfg["Delta_t"]

    lambda1, lambda2, lambda3 = cfg["lambda"]
    cost_I = cfg["cost_I"]
    cost_lambda1 = cfg["cost_lambda"][0]
    kappa = 0.0
    valid_size = cfg["valid_size"]
    n_maxstep = cfg["n_maxstep"]

    # build graphon
    graphon = build_graphon(cfg)

    # SAVE DATA (optional, same as before)
    np.savez('data/data_fbodesolver_params.npz',
             beta=beta, gamma=gamma, kappa=kappa,
             lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
             cost_I=cost_I, cost_lambda1=cost_lambda1,
             m0=m0,
             batch_size=batch_size, valid_size=valid_size, n_maxstep=n_maxstep)

    # SOLVE FBODE
    tf.random.set_seed(n_seed)
    tf.keras.backend.set_floatx('float64')
    print("============ BEGIN SOLVER FBODE ============")

    ode_equation = FBODE.FBODEEquation(beta, gamma, kappa,
                                       lambda1, lambda2, lambda3,
                                       cost_I, cost_lambda1,
                                       g, Delta_t)
    ode_solver = fbodesolver.SolverODE(ode_equation, T, m0, batch_size, valid_size,
                                       n_maxstep, cfg=cfg, graphon=graphon)
    ode_solver.train()

    np.savez('data/data_fbodesolver_solution_final.npz',
             t_path=ode_solver.t_path,
             P_path=ode_solver.P_path,
             U_path=ode_solver.U_path,
             X_path=ode_solver.X_path,
             ALPHA_path=ode_solver.ALPHA_path,
             Z_path=ode_solver.Z_empirical_path,
             loss_history=ode_solver.loss_history)
    print("============ END SOLVER FBODE ============")

if __name__ == '__main__':
    np.random.seed(n_seed)
    main()
