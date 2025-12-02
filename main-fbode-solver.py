
import FBODE as FBODE
import fbodesolver as fbodesolver
import tensorflow as tf
import numpy as np
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

print('\n\ntf.VERSION = ', tf.__version__, '\n\n')
print('\n\ntf.keras.__version__ = ', tf.keras.__version__, '\n\n')

cfg = load_config("config.yaml")
n_seed = cfg["solver"]["n_seed"]

def main():
    cfg = load_config("config.yaml")
    Nstates = cfg["solver"]["Nstates"]
    batch_size  = cfg["solver"]["batch_size"]
    valid_size  = cfg["solver"]["valid_size"]
    n_maxstep   = cfg["solver"]["n_maxstep"]
    
    # Solver params
    I0 = cfg["solver"]["I0"]
    R0 = cfg["solver"]["R0"]
    S0 = 1.0 - I0 - R0
    m0 = [S0, I0, R0]
    T = cfg["solver"]["T"]

    # NN
    lr_boundaries = cfg["solver"]["lr_boundaries"]
    lr_values = cfg["solver"]["lr_values"]
    stdNN=cfg["solver"]["stdNN"]

    # FBODE
    beta        = cfg["solver"]["beta"]
    gamma       = cfg["solver"]["gamma"]
    kappa       = cfg["solver"]["kappa"]
    lambda1     = cfg["solver"]["lambda1"]
    lambda2     = cfg["solver"]["lambda2"]
    lambda3     = cfg["solver"]["lambda3"]
    cost_I      = cfg["solver"]["cost_I"]
    cost_lambda1= cfg["solver"]["cost_lambda1"]
    g           = cfg["solver"]["g"]
    Delta_t     = cfg["solver"]["Delta_t"]


    

    #graphon mode
    graphon_mode = cfg["graphon"]["graphon_mode"]
    graphon_cfg= cfg["graphon"]

    # Display & Archive
    n_displaystep = cfg["solver"]["n_displaystep"]
    n_savetofilestep =cfg["solver"]["n_savetofilestep"]
    

    # SAVE DATA TO FILE
    datafile_name = 'data/data_fbodesolver_params.npz'
    np.savez(datafile_name,
                beta=beta, gamma=gamma, kappa=kappa,
                lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,
                cost_I=cost_I, cost_lambda1=cost_lambda1,
                m0=m0,
                batch_size=batch_size, valid_size=valid_size, n_maxstep=n_maxstep)
    # SOLVE FBODE USING NN
    tf.random.set_seed(n_seed) # ---------- TF2
    tf.keras.backend.set_floatx('float64') # To change all layers to have dtype float64 by default, useful when porting from TF 1.X to TF 2.0  # ---------- TF2
    print("============ BEGIN SOLVER FBODE ============")
    print("================ PARAMETERS ================")
    print("graphon mode:", graphon_mode)
    print("particles:", batch_size, "T:", T, "I0:", I0)
    print("beta:", beta, "gamma:", gamma, "kappa:", kappa)
    print("max_step:", n_maxstep, "Delta_t:", Delta_t)
    ode_equation = FBODE.FBODEEquation(beta, gamma, kappa,lambda1, lambda2, lambda3,cost_I, cost_lambda1,g, Delta_t,graphon_mode = graphon_mode,graphon_cfg  = graphon_cfg)
    ode_solver = fbodesolver.SolverODE(
        equation       = ode_equation,
        T              = T,
        m0             = m0,
        batch_size     = batch_size,
        valid_size     = valid_size,
        n_maxstep      = n_maxstep,
        Nstates        = Nstates,
        n_displaystep  = n_displaystep,
        n_savetofilestep = n_savetofilestep,
        stdNN          = stdNN,
        lr_boundaries  = lr_boundaries,
        lr_values      = lr_values) # ---------- TF2
    # === LOAD LATEST CHECKPOINT IF EXISTS ===
    ode_solver.train()
    # SAVE TO FILE
    print("SAVING FILES...")
    datafile_name = 'data/data_fbodesolver_solution_final.npz'
    np.savez(datafile_name,
             t_path = ode_solver.t_path,
             P_path = ode_solver.P_path,
             U_path = ode_solver.U_path,
             X_path = ode_solver.X_path,
             ALPHA_path = ode_solver.ALPHA_path,
             Z_path = ode_solver.Z_empirical_path,
             loss_history = ode_solver.loss_history)
    print("============ END SOLVER FBODE ============")


if __name__ == '__main__':
    np.random.seed(n_seed)
    main()
