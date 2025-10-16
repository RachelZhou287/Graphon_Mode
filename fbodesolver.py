import time
import tensorflow as tf
import numpy as np
import os
import math


class SolverODE:
    def __init__(self, equation, T, m0, batch_size, valid_size, n_maxstep, cfg=None, graphon=None):  # patched
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
        self.lr_boundaries = [100]  # FOR LRSCHEDULE
        self.lr_values = [1e-2, 1e-2]  # FOR LRSCHEDULE
        self.activation_fn_choice = tf.nn.sigmoid  # tf.nn.leaky_relu

        self.alpha_I_pop = tf.cast(self.equation.lambda2, tf.float64)
        self.beta = self.equation.beta  # Keep as-is, handled in beta_creator
        self.g = self.equation.g
        self.Delta_t = tf.cast(self.equation.Delta_t, tf.float64)
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
        return tf.random.uniform(shape=(1, n_samples), dtype=tf.float64)

    def sample_x0(self, n_samples, id_):
        # build initial state samples [S0, I0, R0, id]
        states = tf.tile(tf.reshape(self.m0, [1, -1]), [n_samples, 1])
        return tf.concat([states, tf.transpose(id_)], axis=1)

    def build_weight_matrix(self, ids):
        """
        Build adjacency-like weight matrix W[i,j] = graphon(id_i, id_j).
        ids: shape (n_samples,) in [0,1]
        """
        ids_np = ids.numpy()
        n = len(ids_np)
        W_np = np.zeros((n, n))
        for i in range(n):
            W_np[i, :] = self.graphon_vec(ids_np[i], ids_np)
        return tf.constant(W_np, dtype=tf.float64)

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
            return tf.fill([n_samples, 1], tf.cast(self.beta, tf.float64))

        # Convert beta into tensorflow tensor
        beta_array = tf.convert_to_tensor(self.beta, dtype=tf.float64)
        n_groups = tf.shape(beta_array)[0]

        # --- Group boundaries ---
        if "piecewise" in self.cfg and "groups" in self.cfg["piecewise"]:
            groups = tf.convert_to_tensor(self.cfg["piecewise"]["groups"], dtype=tf.float64)
            boundaries = tf.cumsum(groups)  # e.g., [0.4, 0.75, 1.0] for 3 groups
        else:
            # If no piecewise groups defined, split evenly
            boundaries = tf.linspace(tf.constant(0.0, dtype=tf.float64), tf.constant(1.0, dtype=tf.float64), n_groups + 1)[1:]

        # --- Assign beta to each agent ---
        ids_flat = tf.reshape(id_, [-1])
        group_idx = tf.searchsorted(boundaries, ids_flat, side='right')
        group_idx = tf.minimum(group_idx, n_groups - 1)
        beta_vector = tf.gather(beta_array, group_idx)
        return tf.reshape(beta_vector, [n_samples, 1])


    # ========== MODEL

    class MyModelU0(tf.keras.Model):
        def __init__(self, Nstates):
            super().__init__()
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
        self.X0 = self.sample_x0(n_samples, self.id_)

        ids_flat = tf.reshape(self.id_, [-1])
        self.W = self.build_weight_matrix(ids_flat)
        self.beta_vector = self.beta_creator(n_samples, self.id_)

        # BUILD THE ODE SDE DYNAMICS
        t = 0.0
        X = self.X0
        self.input = tf.transpose(self.id_)
        U = self.model_u0.call_u0(self.input)
        Z_empirical = tf.reshape(tf.matmul(self.W, tf.reshape(X[:, 1], [n_samples, 1])) * self.alpha_I_pop / tf.cast(n_samples, tf.float64), [n_samples, 1])
        ALPHA = self.equation.optimal_ALPHA(U, Z_empirical, self.beta_vector)
        P = tf.tile(tf.reshape(self.m0, [1, self.Nstates]), [n_samples, 1])

        # STORE SOLUTION
        self.U_path = tf.expand_dims(U, axis=1)
        self.P_path = tf.expand_dims(P, axis=1)
        self.X_path = tf.expand_dims(X, axis=1)
        self.ALPHA_path = tf.expand_dims(ALPHA, axis=1)
        self.t_path = tf.reshape(tf.cast(t, tf.float64), (1, 1))
        self.Z_empirical_path = tf.expand_dims(Z_empirical, axis=1)

        # DYNAMICS FOR P AND U
        P_next = P + self.equation.driver_P(Z_empirical, P, ALPHA, self.beta_vector) * self.Delta_t
        U_next = U + self.equation.driver_U(Z_empirical, U, ALPHA, self.beta_vector) * self.Delta_t
        X_next = tf.concat([P_next, tf.transpose(self.id_)], axis=1)

        # UPDATES
        P = P_next
        U = U_next
        X = X_next
        Z_empirical = tf.reshape(tf.matmul(self.W, tf.reshape(X[:, 1], [n_samples, 1])) * self.alpha_I_pop / tf.cast(n_samples, tf.float64), [n_samples, 1])

        # LOOP IN TIME
        for i_t in range(1, self.Nmax_iter + 1):
            if (tf.reduce_mean(X[:, 1]).numpy() < 1.e-3):  # TO AVOID NAN
                # print("INNER P_empirical[1]<1.e-3 = ", tf.reduce_mean(X[:, 1]).numpy(), "\t t = ", t)
                break
            t = t + self.Delta_t.numpy()
            ALPHA = self.equation.optimal_ALPHA(U, Z_empirical, self.beta_vector)

            if (t < self.T.numpy()):
                P_next = P + self.equation.driver_P(Z_empirical, P, ALPHA, self.beta_vector) * self.Delta_t
                U_next = U + self.equation.driver_U(Z_empirical, U, ALPHA, self.beta_vector) * self.Delta_t
                X_next = tf.concat([P_next, tf.transpose(self.id_)], axis=1)

                # STORE SOLUTION
                self.t_path = tf.concat([self.t_path, tf.reshape(tf.cast(t, tf.float64), (1, 1))], axis=1)
                self.P_path = tf.concat([self.P_path, tf.expand_dims(P, axis=1)], axis=1)
                self.U_path = tf.concat([self.U_path, tf.expand_dims(U, axis=1)], axis=1)
                self.X_path = tf.concat([self.X_path, tf.expand_dims(X, axis=1)], axis=1)
                self.ALPHA_path = tf.concat([self.ALPHA_path, tf.expand_dims(ALPHA, axis=1)], axis=1)
                self.Z_empirical_path = tf.concat([self.Z_empirical_path, tf.expand_dims(Z_empirical, axis=1)], axis=1)

                # UPDATES
                P = P_next
                U = U_next
                X = X_next
                Z_empirical = tf.reshape(tf.matmul(self.W, tf.reshape(X[:, 1], [n_samples, 1])) * self.alpha_I_pop / tf.cast(n_samples, tf.float64), [n_samples, 1])
            else:
                break

        # COMPUTE ERROR
        target = tf.zeros([tf.shape(U)[0], self.Nstates], dtype=tf.float64)
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
        self.loss_history.append(temp_loss.numpy())
        step = 0
        print("step: %5u, loss: %.4e " % (step, temp_loss) +
              "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass))
        file = open("res-console.txt", "a")
        file.write("step: %5u, loss: %.4e " % (0, temp_loss) +
                   "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass) + "\n")
        file.close()
        datafile_name = 'data/data_fbodesolver_solution_iter{}.npz'.format(step)
        np.savez(datafile_name,
                 t_path=self.t_path.numpy(),
                 P_path=self.P_path.numpy(),
                 U_path=self.U_path.numpy(),
                 X_path=self.X_path.numpy(),
                 ALPHA_path=self.ALPHA_path.numpy(),
                 Z_path=self.Z_empirical_path.numpy(),
                 loss_history=np.array(self.loss_history))

        # BEGIN SGD ITERATION
        for step in range(1, self.n_maxstep + 1):
            with tf.GradientTape() as tape:
                _ = self.forward_pass(self.batch_size)
                curr_loss = self.loss
            grads = tape.gradient(curr_loss, self.model_u0.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.model_u0.trainable_variables))

            if step == 1 or step % self.n_displaystep == 0:
                _ = self.forward_pass(self.valid_size)
                temp_loss = self.loss
                self.loss_history.append(temp_loss.numpy())
                print("step: %5u, loss: %.4e " % (step, temp_loss) +
                      "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass))
                file = open("res-console.txt", "a")
                file.write("step: %5u, loss: %.4e " % (step, temp_loss) +
                           "runtime: %4u s" % (time.time() - start_time + self.time_forward_pass) + "\n")
                file.close()
                if (step == 1 or step % self.n_savetofilestep == 0):
                    datafile_name = 'data/data_fbodesolver_solution_iter{}.npz'.format(step)
                    np.savez(datafile_name,
                             t_path=self.t_path.numpy(),
                             P_path=self.P_path.numpy(),
                             U_path=self.U_path.numpy(),
                             X_path=self.X_path.numpy(),
                             ALPHA_path=self.ALPHA_path.numpy(),
                             Z_path=self.Z_empirical_path.numpy(),
                             loss_history=np.array(self.loss_history))
                    checkpoint.save(file_prefix=checkpoint_prefix)
            elif (step % self.n_savetofilestep == 0):
                print("SAVING TO DATA FILE...")
                _ = self.forward_pass(self.valid_size)
                temp_loss = self.loss
                self.loss_history.append(temp_loss.numpy())
                datafile_name = 'data/data_fbodesolver_solution_iter{}.npz'.format(step)
                np.savez(datafile_name,
                         t_path=self.t_path.numpy(),
                         P_path=self.P_path.numpy(),
                         U_path=self.U_path.numpy(),
                         X_path=self.X_path.numpy(),
                         ALPHA_path=self.ALPHA_path.numpy(),
                         Z_path=self.Z_empirical_path.numpy(),
                         loss_history=np.array(self.loss_history))

        datafile_name = 'data/data_fbodesolver_solution_iter-final.npz'
        np.savez(datafile_name,
                 t_path=self.t_path.numpy(),
                 P_path=self.P_path.numpy(),
                 U_path=self.U_path.numpy(),
                 X_path=self.X_path.numpy(),
                 ALPHA_path=self.ALPHA_path.numpy(),
                 Z_path=self.Z_empirical_path.numpy(),
                 loss_history=np.array(self.loss_history))
        end_time = time.time()
        print("running time: %.3f s" % (end_time - self.time_init))
        print('========== END TRAINING ==========')