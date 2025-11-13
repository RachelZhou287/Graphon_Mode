# FBODE.py
import tensorflow as tf
import numpy as np




class FBODEEquation(object):
    def __init__(self, beta, gamma, kappa, lambda1, lambda2, lambda3, cost_I, cost_lambda1, g, Delta_t,graphon_mode,graphon_cfg):
        # parameters of the MFG
        self.beta = beta
        self.gamma = gamma
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.cost_I = cost_I # penalty for being in I
        self.cost_lambda1 = cost_lambda1
        self.kappa = kappa
        self.g = g
        self.Delta_t = Delta_t
        self.graphon_mode = graphon_mode
        self.graphon_cfg = graphon_cfg


    def driver_P(self, Z_empirical, P, ALPHA, beta_vector):
        n_samples =    tf.shape(P)[0]
        Nstates =      tf.shape(P)[1]
        driver =       tf.zeros((n_samples, Nstates), dtype=tf.float64)
        S = tf.reshape(P[:,0], (n_samples,1))
        I = tf.reshape(P[:,1], (n_samples,1))
        R = tf.reshape(P[:,2], (n_samples,1))
        
        dS = -beta_vector * ALPHA * Z_empirical * S + self.kappa * R
        dI =  beta_vector * ALPHA * Z_empirical * S - self.gamma * I
        dR =  self.gamma * I - self.kappa * R
        
        driver = tf.concat([dS, dI, dR], axis=1)
        return driver


    def driver_U(self, Z_empirical, U, ALPHA, beta_vector):
        n_samples =    tf.shape(U)[0]
        Nstates =      tf.shape(U)[1]

        U0 = tf.reshape(U[:, 0], (n_samples, 1))
        U1 = tf.reshape(U[:, 1], (n_samples, 1))
        U2 = tf.reshape(U[:, 2], (n_samples, 1))

        dU0 = (beta_vector * ALPHA * Z_empirical * (U0 - U1)
               - 0.5 * self.cost_lambda1 * (self.lambda1 - ALPHA)**2)

        dU1 = self.gamma * (U1 - U2) - self.cost_I
        dU2 = self.kappa * (U2 - U0)
        driver= tf.concat([dU0, dU1, dU2], axis=1)
        return driver

    def optimal_ALPHA(self, U, Z_empirical, beta_vector):
        n_samples = tf.shape(U)[0]

        U0 = tf.reshape(U[:, 0], (n_samples, 1))
        U1 = tf.reshape(U[:, 1], (n_samples, 1))

        return self.lambda1 + 0.5 * (beta_vector / self.cost_lambda1) * \
               Z_empirical * (U0 - U1)
