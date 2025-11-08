import time
import tensorflow as tf
import numpy as np
import os
import math

# fbodesolver.py
tf.keras.backend.set_floatx('float64')  # Enforce 64-bit precision globally


class SolverODE:
    def __init__(self, equation, T, m0, batch_size, valid_size, n_maxstep, cfg=None, graphon=None):
        # initial condition and horizon
        self.m0 = tf.convert_to_tensor(m0, dtype=tf.float64)
        self.T = tf.cast(T, tf.float64)

        # equation (of ode equation class)
        self.equation = equation

        # parameters for neural network and gradient descent
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.n_maxstep = n_maxstep
        self.n_displaystep = 5
        self.n_savetofilestep = 10
        self.stdNN = 1.e-2
        self.lr_boundaries = [500]
        self.lr_values = [1e-2, 1e-3] 
        self.activation_fn_choice = tf.nn.sigmoid

        self.alpha_I_pop = tf.cast(self.equation.lambda2, tf.float64)
        self.beta = self.equation.beta
        self.g = tf.cast(self.equation.g, tf.float64)
        self.Delta_t = tf.cast(self.equation.Delta_t, tf.float64)
        self.Nmax_iter = int(math.ceil(self.T / self.Delta_t))
        self.Nstates = 3  # S I R

        # --- graphon setup ---
        self.cfg = cfg or {}
        if graphon is not None:
            self.graphon = graphon
        else:
            # fallback: construct from mode
            mode = self.cfg.get("graphon_mode", "constant")
            if mode == "powerlaw":
                g_exp = tf.cast(self.cfg.get("g", -0.3), tf.float64)

                def powerlaw_graphon(x, y):
                    eps = tf.constant(1e-8, tf.float64)
                    return tf.pow(tf.maximum(x, eps) * tf.maximum(y, eps), g_exp)

                self.graphon = powerlaw_graphon
            elif mode == "constant":
                p = self.cfg.get("p", 1.0)
                self.graphon = lambda x, y: tf.cast(p, tf.float64)
            else:
                p = self.cfg.get("p", 1.0)
                self.graphon = lambda x, y: tf.cast(p, tf.float64)

        self.graphon_vec = np.vectorize(
            lambda x, y: float(self.graphon(tf.constant(x, tf.float64), tf.constant(y, tf.float64))),
            otypes=[float]
        )

        # neural net
        self.model_u0 = self.MyModelU0(self.Nstates)

        # timing
        self.time_init = time.time()

    def id_creator(self, n_samples):
        return tf.random.uniform(shape=(1, n_samples), dtype=tf.float64)

    def sample_x0(self, n_samples, id_):
        # initial state samples [S0, I0, R0, id]
        states = tf.tile(tf.reshape(self.m0, [1, -1]), [n_samples, 1])
        return tf.concat([states, tf.transpose(id_)], axis=1)

    #def build_weight_matrix(self, ids):
        #"""
        #Stable adjacency matrix W[i,j] = graphon(id_i, id_j).
        #Uses float64 and supports power-law mode safely.
        #"""
        #ids = tf.cast(ids, tf.float64)
        #n = tf.shape(ids)[0]
        #ids_col = tf.reshape(ids, [n, 1])
        #ids_row = tf.reshape(ids, [1, n])
        #Xi = tf.tile(ids_col, [1, n])
        #Xj = tf.tile(ids_row, [n, 1])

        # compute using self.graphon in TF directly
        #g_fn = self.graphon
        #if self.cfg.get("graphon_mode", "constant") == "powerlaw":
            #eps = tf.constant(1e-9, tf.float64)
           # W = tf.pow(tf.maximum(Xi, eps) * tf.maximum(Xj, eps), self.g)
        
        #else:
            #W = g_fn(Xi, Xj)

        #return tf.cast(W, tf.float64)

    def build_weight_matrix(self, ids):
        ids = tf.cast(ids, tf.float64)
        n = tf.shape(ids)[0]
        ids_col = tf.reshape(ids, [n, 1])
        ids_row = tf.reshape(ids, [1, n])
        Xi = tf.tile(ids_col, [1, n])
        Xj = tf.tile(ids_row, [n, 1])
    
        mode = self.cfg.get("graphon_mode", "constant")
        if mode == "powerlaw":
            g = tf.cast(self.cfg.get("g", 0.3), tf.float64)
            eps = tf.constant(1e-8, tf.float64)
            W = tf.pow(tf.maximum(Xi, eps) * tf.maximum(Xj, eps), g)
        elif mode == "piecewise":
            from graphon_utils import build_graphon
            w_fn = build_graphon(self.cfg)
            W = w_fn(Xi, Xj)
        else:
            p = tf.cast(self.cfg.get("p", 1.0), tf.float64)
            W = tf.fill([n, n], p)
        return W



    def beta_creator(self, n_samples, id_):
        """
        Assign group-specific infection rates (beta) to each agent.
        """
        if isinstance(self.beta, (float, int)):
            return tf.fill([n_samples, 1], tf.cast(self.beta, tf.float64))

        beta_array = tf.convert_to_tensor(self.beta, dtype=tf.float64)
        n_groups = tf.shape(beta_array)[0]

        if "piecewise" in self.cfg and "groups" in self.cfg["piecewise"]:
            groups = tf.convert_to_tensor(self.cfg["piecewise"]["groups"], dtype=tf.float64)
            boundaries = tf.cumsum(groups)
        else:
            boundaries = tf.linspace(tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64), n_groups + 1)[1:]

        ids_flat = tf.reshape(id_, [-1])
        group_idx = tf.searchsorted(boundaries, ids_flat, side='right')
        group_idx = tf.minimum(group_idx, n_groups - 1)
        beta_vector = tf.gather(beta_array, group_idx)
        return tf.reshape(beta_vector, [n_samples, 1])

    # ===================== MODEL =====================

    class MyModelU0(tf.keras.Model):
        def __init__(self, Nstates):
            super().__init__()
            init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5)
            self.layer1_u0 = tf.keras.layers.Dense(100, activation='sigmoid', kernel_initializer=init, bias_initializer='zeros', dtype=tf.float64)
            self.layer2_u0 = tf.keras.layers.Dense(50, activation='sigmoid', kernel_initializer=init, bias_initializer='zeros', dtype=tf.float64)
            self.layer3_u0 = tf.keras.layers.Dense(50, activation='sigmoid', kernel_initializer=init, bias_initializer='zeros', dtype=tf.float64)
            self.layer4_u0 = tf.keras.layers.Dense(50, activation='sigmoid', kernel_initializer=init, bias_initializer='zeros', dtype=tf.float64)
            self.layer5_u0 = tf.keras.layers.Dense(Nstates, activation='sigmoid', dtype=tf.float64)

        def call_u0(self, input):
            x = tf.cast(input, tf.float64)
            x = self.layer1_u0(x)
            x = self.layer2_u0(x)
            x = self.layer3_u0(x)
            x = self.layer4_u0(x)
            x = 15.0 * self.layer5_u0(x) #15.0 * self.layer5_u0(x)
            return x

    # ===================== FORWARD PASS =====================

    def forward_pass(self, n_samples):
        start_time = time.time()

        self.id_ = self.id_creator(n_samples)
        self.X0 = tf.cast(self.sample_x0(n_samples, self.id_), tf.float64)
        ids_flat = tf.reshape(self.id_, [-1])
        self.W = tf.cast(self.build_weight_matrix(ids_flat), tf.float64)
        self.beta_vector = tf.cast(self.beta_creator(n_samples, self.id_), tf.float64)

        t = tf.constant(0.0, tf.float64)
        X = self.X0
        self.input = tf.transpose(self.id_)
        U = self.model_u0.call_u0(self.input)

        Z_empirical = tf.reshape(tf.matmul(self.W, tf.reshape(X[:, 1], [n_samples, 1])) *self.alpha_I_pop /tf.cast(n_samples, tf.float64),[n_samples, 1])
        
        ALPHA = self.equation.optimal_ALPHA(U, Z_empirical, self.beta_vector)
        P = tf.tile(tf.reshape(self.m0, [1, self.Nstates]), [n_samples, 1])

        # STORE
        self.U_path = tf.expand_dims(U, axis=1)
        self.P_path = tf.expand_dims(P, axis=1)
        self.X_path = tf.expand_dims(X, axis=1)
        self.ALPHA_path = tf.expand_dims(ALPHA, axis=1)
        self.t_path = tf.reshape(t, (1, 1))
        self.Z_empirical_path = tf.expand_dims(Z_empirical, axis=1)

        # ONE STEP
        P_next = P + self.equation.driver_P(Z_empirical, P, ALPHA, self.beta_vector) * self.Delta_t
        # U_next = U + self.equation.driver_U(Z_empirical, U, ALPHA, self.beta_vector) * self.Delta_t
        U_next = U + self.equation.driver_U(Z_empirical, U, ALPHA, self.beta_vector) * self.Delta_t
        X_next = tf.concat([P_next, tf.transpose(self.id_)], axis=1)
        P, U, X = P_next, U_next, X_next
        Z_empirical = tf.reshape(np.matmul(self.W, X[:,1]) * self.alpha_I_pop / n_samples, [n_samples,1])
        # LOOP
        for _ in range(1, self.Nmax_iter + 1):
            mean_inf = tf.reduce_mean(X[:, 1])
            if mean_inf < 1e-3:
                break
            t = t + self.Delta_t
            ALPHA = self.equation.optimal_ALPHA(U, Z_empirical, self.beta_vector)

            if t < self.T:
                P_next = P + self.equation.driver_P(Z_empirical, P, ALPHA, self.beta_vector) * self.Delta_t
                U_next = U + self.equation.driver_U(Z_empirical, U, ALPHA, self.beta_vector) * self.Delta_t
                X_next = tf.concat([P_next, tf.transpose(self.id_)], axis=1)

                self.t_path = tf.concat([self.t_path, tf.reshape(t, (1, 1))], axis=1)
                self.P_path = tf.concat([self.P_path, tf.expand_dims(P, axis=1)], axis=1)
                self.U_path = tf.concat([self.U_path, tf.expand_dims(U, axis=1)], axis=1)
                self.X_path = tf.concat([self.X_path, tf.expand_dims(X, axis=1)], axis=1)
                self.ALPHA_path = tf.concat([self.ALPHA_path, tf.expand_dims(ALPHA, axis=1)], axis=1)
                self.Z_empirical_path = tf.concat([self.Z_empirical_path, tf.expand_dims(Z_empirical, axis=1)], axis=1)

                P, U, X = P_next, U_next, X_next
                Z_empirical = tf.reshape(tf.matmul(self.W, tf.reshape(X[:, 1], [n_samples, 1])) *self.alpha_I_pop, [n_samples, 1])
            else:
                break

        # target = tf.zeros([tf.shape(U)[0], self.Nstates], dtype=tf.float64)
        target = tf.zeros_like(U)
        error = U - target
        self.loss = tf.reduce_mean(tf.square(error))
        self.time_forward_pass = time.time() - start_time
        return self.loss

    # ===================== TRAIN =====================

    def train(self):
        print('========== START TRAINING ==========')
        start_time = time.time()
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.lr_boundaries, self.lr_values)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        checkpoint_directory = "checkpoints/"
        os.makedirs(checkpoint_directory, exist_ok=True)
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer,
            model=self.model_u0,
            optimizer_step=tf.compat.v1.train.get_or_create_global_step()
        )
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

        self.loss_history = []

        _ = self.forward_pass(self.valid_size)
        temp_loss = self.loss
        self.loss_history.append(temp_loss.numpy())
        step = 0
        print(f"step: {step:5d}, loss: {temp_loss:.4e} runtime: {int(time.time() - start_time + self.time_forward_pass):4d}s")
        with open("res-console.txt", "a") as f:
            f.write(f"step: {step:5d}, loss: {temp_loss:.4e} runtime: {int(time.time() - start_time + self.time_forward_pass):4d}s\n")

        os.makedirs("data", exist_ok=True)
        np.savez(f'data/data_fbodesolver_solution_iter{step}.npz',
                 t_path=self.t_path.numpy(), P_path=self.P_path.numpy(),
                 U_path=self.U_path.numpy(), X_path=self.X_path.numpy(),
                 ALPHA_path=self.ALPHA_path.numpy(), Z_path=self.Z_empirical_path.numpy(),
                 loss_history=np.array(self.loss_history))

        # MAIN TRAIN LOOP
        for step in range(1, self.n_maxstep + 1):
            with tf.GradientTape() as tape:
                loss = self.forward_pass(self.batch_size)
                #_ = self.forward_pass(self.batch_size)
            grads = tape.gradient(loss, self.model_u0.trainable_variables)
            # --- DEBUG: check gradients ---
            grad_status = [g is None for g in grads]
            grad_max = [tf.reduce_max(tf.abs(g)).numpy() if g is not None else None for g in grads]
            #print(f"[DEBUG] Step {step} grad None flags: {grad_status}")
            #print(f"[DEBUG] Step {step} grad max abs values: {grad_max}")

            optimizer.apply_gradients(zip(grads, self.model_u0.trainable_variables))

            if step == 1 or step % self.n_displaystep == 0:
                _ = self.forward_pass(self.valid_size)
                temp_loss = self.loss
                self.loss_history.append(temp_loss.numpy())
                print(f"step: {step:5d}, loss: {temp_loss:.4e} runtime: {int(time.time() - start_time + self.time_forward_pass):4d}s")
                with open("res-console.txt", "a") as f:
                    f.write(f"step: {step:5d}, loss: {temp_loss:.4e} runtime: {int(time.time() - start_time + self.time_forward_pass):4d}s\n")
                if step == 1 or step % self.n_savetofilestep == 0:
                    np.savez(f'data/data_fbodesolver_solution_iter{step}.npz',
                             t_path=self.t_path.numpy(), P_path=self.P_path.numpy(),
                             U_path=self.U_path.numpy(), X_path=self.X_path.numpy(),
                             ALPHA_path=self.ALPHA_path.numpy(), Z_path=self.Z_empirical_path.numpy(),
                             loss_history=np.array(self.loss_history))
                    checkpoint.save(file_prefix=checkpoint_prefix)
            elif step % self.n_savetofilestep == 0:
                print("SAVING TO DATA FILE...")
                _ = self.forward_pass(self.valid_size)
                temp_loss = self.loss
                self.loss_history.append(temp_loss.numpy())
                np.savez(f'data/data_fbodesolver_solution_iter{step}.npz',
                         t_path=self.t_path.numpy(), P_path=self.P_path.numpy(),
                         U_path=self.U_path.numpy(), X_path=self.X_path.numpy(),
                         ALPHA_path=self.ALPHA_path.numpy(), Z_path=self.Z_empirical_path.numpy(),
                         loss_history=np.array(self.loss_history))

        np.savez('data/data_fbodesolver_solution_iter-final.npz',
                 t_path=self.t_path.numpy(), P_path=self.P_path.numpy(),
                 U_path=self.U_path.numpy(), X_path=self.X_path.numpy(),
                 ALPHA_path=self.ALPHA_path.numpy(), Z_path=self.Z_empirical_path.numpy(),
                 loss_history=np.array(self.loss_history))
        print(f"running time: {time.time() - self.time_init:.3f}s")
        print('========== END TRAINING ==========')
