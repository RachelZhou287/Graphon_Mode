import time
import tensorflow as tf
import numpy as np
import os
import math


class SolverODE:
    def __init__(self, equation, T, m0, batch_size, valid_size, n_maxstep, cfg=None, graphon=None):  # patched
        # initial condition and horizon
        self.m0 = m0
        self.T = T

        # equation (of ode equation class)
        self.equation = equation

        # parameters for neural network and gradient descent
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.n_maxstep = n_maxstep
        self.n_displaystep = 5
        self.n_savetofilestep = 10
        self.stdNN = 1.e-2
        self.lr_boundaries = [100]  # FOR LRSCHEDULE
        self.lr_values = [1e-2, 1e-2]  # FOR LRSCHEDULE
        self.activation_fn_choice = tf.nn.sigmoid  # tf.nn.leaky_relu

        self.alpha_I_pop = self.equation.lambda2
        self.beta = self.equation.beta
        self.g = self.equation.g
        self.Delta_t = self.equation.Delta_t
        self.Nmax_iter = int(math.ceil(self.T / self.Delta_t))
        self.Nstates = 3  # S I R for now

        # --- graphon setup (new) ---
        self.cfg = cfg or {}
        if graphon is not None:
            self.graphon = graphon
        else:
            # fallback to constant if nothing provided
            p = self.cfg.get("p", 1.0) if isinstance(self.cfg, dict) else 1.0
            self.graphon = lambda x, y: p
        # vectorized version (works with numpy arrays)
        self.graphon_vec = np.vectorize(self.graphon, otypes=[float])

        # neural net
        self.model_u0 = self.MyModelU0(self.Nstates)

        # timing
        self.time_init = time.time()

    def id_creator(self, n_samples):
        return np.random.uniform(size=(1, n_samples))

    def sample_x0(self, n_samples, id_):
        # build initial state samples [S0, I0, R0, id]
        return np.concatenate((np.tile(self.m0, (n_samples, 1)), id_.T), axis=1)

    def build_weight_matrix(self, ids):
        """
        Build adjacency-like weight matrix W[i,j] = graphon(id_i, id_j).
        ids: shape (n_samples,) in [0,1]
        """
        n = len(ids)
        W = np.zeros((n, n))
        for i in range(n):
            W[i, :] = self.graphon_vec(ids[i], ids)
        return W

    def beta_creator(self, n_samples, id_):
        """
        Assign group-specific infection rates (beta) to each agent.

        Rules:
        - If beta is a float, broadcast to all agents.
        - If beta is a list/array with length > 1:
            * If 'piecewise' groups are defined in cfg, assign based on those.
            * Otherwise, split [0,1] interval evenly into len(beta) groups.
        """
        if isinstance(self.beta, (float, int)):
            # Single global infection rate
            return np.full((n_samples, 1), float(self.beta))

        # Convert beta into numpy array
        beta_array = np.array(self.beta, dtype=float)
        n_groups = beta_array.size

        # --- Group boundaries ---
        if "piecewise" in self.cfg and "groups" in self.cfg["piecewise"]:
            groups = np.array(self.cfg["piecewise"]["groups"], dtype=float)
            boundaries = np.cumsum(groups)  # e.g., [0.4, 0.75, 1.0] for 3 groups
        else:
            # If no piecewise groups defined, split evenly
            boundaries = np.linspace(0, 1, n_groups + 1)[1:]

        # --- Assign beta to each agent ---
        beta_vector = np.zeros((n_samples, 1))
        ids = id_.flatten()
        for i in range(n_samples):
            group_idx = np.searchsorted(boundaries, ids[i], side="right")
            group_idx = min(group_idx, n_groups - 1)  # clamp to last group
            beta_vector[i] = beta_array[group_idx]

        return beta_vector


    # ========== MODEL

    class MyModelU0(tf.keras.Model):
        def __init__(self, Nstates):
            super(tf.keras.Model, self).__init__()
            self.layer1_u0 = tf.keras.layers.Dense(
                units=100, activation='sigmoid',
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),
                bias_initializer='zeros')
            self.layer2_u0 = tf.keras.layers.Dense(
                units=50, activation='sigmoid',
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),
                bias_initializer='zeros')
            self.layer3_u0 = tf.keras.layers.Dense(
                units=50, activation='sigmoid',
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),
                bias_initializer='zeros')
            self.layer4_u0 = tf.keras.layers.Dense(
                units=50, activation='sigmoid',
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),
                bias_initializer='zeros')
            self.layer5_u0 = tf.keras.layers.Dense(
                units=Nstates, activation='sigmoid')

        def call_u0(self, input):
            result = self.layer1_u0(input)
            result = self.layer2_u0(result)
            result = self.layer3_u0(result)
            result = self.layer4_u0(result)
            result = 15 * self.layer5_u0(result)
            return result

    def forward_pass(self, n_samples):
        start_time = time.time()
        # SAMPLE INITIAL POINTS
        self.id_ = self.id_creator(n_samples)  # shape (1, n_samples)
        self.X0 = tf.cast(self.sample_x0(n_samples, self.id_), tf.float64)

        ids_flat = self.id_.flatten()
        self.W = self.build_weight_matrix(ids_flat)
        self.beta_vector = self.beta_creator(n_samples, self.id_)

        # BUILD THE ODE SDE DYNAMICS
        t = 0.0
        X = self.X0
        self.input = tf.cast(self.id_.T, tf.float64)
        U = self.model_u0.call_u0(self.input)
        Z_empirical = tf.reshape(np.matmul(self.W, X[:, 1]) * self.alpha_I_pop / n_samples, [n_samples, 1])
        ALPHA = self.equation.optimal_ALPHA(U, Z_empirical, self.beta_vector)
        P = tf.cast(np.tile(self.m0, (n_samples, 1)), tf.float64)

        # STORE SOLUTION
        self.U_path = U
        self.P_path = P
        self.X_path = X
        self.ALPHA_path = ALPHA
        self.t_path = tf.reshape(tf.cast(t, tf.float64), (1, 1))
        self.Z_empirical_path = Z_empirical

        # DYNAMICS FOR P AND U
        P_next = P + self.equation.driver_P(Z_empirical, P, ALPHA, self.beta_vector) * self.Delta_t
        U_next = U + self.equation.driver_U(Z_empirical, U, ALPHA, self.beta_vector) * self.Delta_t
        X_next = np.concatenate((P_next, self.id_.T), axis=1)

        # UPDATES
        P = P_next
        U = U_next
        X = X_next
        Z_empirical = tf.reshape(np.matmul(self.W, X[:, 1]) * self.alpha_I_pop / n_samples, [n_samples, 1])

        # LOOP IN TIME
        for i_t in range(1, self.Nmax_iter + 1):
            if (np.mean(X[:, 1]) < 1.e-3):  # TO AVOID NAN
                print("INNER P_empirical[1]<1.e-3 = ", np.mean(X[:, 1]), "\t t = ", t)
                break
            t = t + self.Delta_t
            ALPHA = self.equation.optimal_ALPHA(U, Z_empirical, self.beta_vector)

            if (t < self.T):
                P_next = P + self.equation.driver_P(Z_empirical, P, ALPHA, self.beta_vector) * self.Delta_t
                U_next = U + self.equation.driver_U(Z_empirical, U, ALPHA, self.beta_vector) * self.Delta_t
                X_next = np.concatenate((P_next, self.id_.T), axis=1)

                # STORE SOLUTION
                self.t_path = tf.concat([self.t_path, tf.reshape(tf.cast(t, tf.float64), (1, 1))], axis=1)
                self.P_path = tf.concat([self.P_path, P], axis=1)
                self.U_path = tf.concat([self.U_path, U], axis=1)
                self.X_path = tf.concat([self.X_path, X], axis=1)
                self.ALPHA_path = tf.concat([self.ALPHA_path, ALPHA], axis=1)
                self.Z_empirical_path = tf.concat([self.Z_empirical_path, Z_empirical], axis=1)

                # UPDATES
                P = P_next
                U = U_next
                X = X_next
                Z_empirical = tf.reshape(np.matmul(self.W, X[:, 1]) * self.alpha_I_pop / n_samples, [n_samples, 1])
            else:
                break

        # COMPUTE ERROR
        target = tf.zeros((1, self.Nstates), dtype=tf.float64)
        error = U - target
        self.loss = tf.reduce_sum(tf.reduce_mean(error**2, axis=0))
        self.time_forward_pass = time.time() - start_time

    def train(self):
        print('========== START TRAINING ==========')
        start_time = time.time()
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.lr_boundaries, self.lr_values)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        checkpoint_directory = "checkpoints/"
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         model=self.model_u0,
                                         optimizer_step=tf.compat.v1.train.get_or_create_global_step())

        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

        self.loss_history = []

        # INITIALIZATION
        _ = self.forward_pass(self.valid_size)
        temp_loss = self.loss
        self.loss_history.append(temp_loss)
        step = 0
        print("step: %5u, loss: %.4e " % (step, temp_loss) +
              "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass))
        file = open("res-console.txt", "a")
        file.write("step: %5u, loss: %.4e " % (0, temp_loss) +
                   "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass) + "\n")
        file.close()
        datafile_name = 'data/data_fbodesolver_solution_iter{}.npz'.format(step)
        np.savez(datafile_name,
                 t_path=self.t_path,
                 P_path=self.P_path,
                 U_path=self.U_path,
                 X_path=self.X_path,
                 ALPHA_path=self.ALPHA_path,
                 Z_path=self.Z_empirical_path,
                 loss_history=self.loss_history)

        # BEGIN SGD ITERATION
        for step in range(1, self.n_maxstep + 1):
            with tf.GradientTape() as tape:
                _ = self.forward_pass(self.batch_size)
                curr_loss = self.loss
            grads = tape.gradient(curr_loss, self.model_u0.variables)
            optimizer.apply_gradients(zip(grads, self.model_u0.variables))

            if step == 1 or step % self.n_displaystep == 0:
                _ = self.forward_pass(self.valid_size)
                temp_loss = self.loss
                self.loss_history.append(temp_loss)
                print("step: %5u, loss: %.4e " % (step, temp_loss) +
                      "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass))
                file = open("res-console.txt", "a")
                file.write("step: %5u, loss: %.4e " % (step, temp_loss) +
                           "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass) + "\n")
                file.close()
                if (step == 1 or step % self.n_savetofilestep == 0):
                    datafile_name = 'data/data_fbodesolver_solution_iter{}.npz'.format(step)
                    np.savez(datafile_name,
                             t_path=self.t_path,
                             P_path=self.P_path,
                             U_path=self.U_path,
                             X_path=self.X_path,
                             ALPHA_path=self.ALPHA_path,
                             Z_path=self.Z_empirical_path,
                             loss_history=self.loss_history)
                    checkpoint.save(file_prefix=checkpoint_prefix)
            elif (step % self.n_savetofilestep == 0):
                print("SAVING TO DATA FILE...")
                _ = self.forward_pass(self.valid_size)
                temp_loss = self.loss
                self.loss_history.append(temp_loss)
                datafile_name = 'data/data_fbodesolver_solution_iter{}.npz'.format(step)
                np.savez(datafile_name,
                         t_path=self.t_path,
                         P_path=self.P_path,
                         U_path=self.U_path,
                         X_path=self.X_path,
                         ALPHA_path=self.ALPHA_path,
                         Z_path=self.Z_empirical_path,
                         loss_history=self.loss_history)

        datafile_name = 'data/data_fbodesolver_solution_iter-final.npz'
        np.savez(datafile_name,
                 t_path=self.t_path,
                 P_path=self.P_path,
                 U_path=self.U_path,
                 X_path=self.X_path,
                 ALPHA_path=self.ALPHA_path,
                 Z_path=self.Z_empirical_path,
                 loss_history=self.loss_history)
        end_time = time.time()
        print("running time: %.3f s" % (end_time - self.time_init))
        print('========== END TRAINING ==========')
