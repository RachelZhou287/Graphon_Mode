import tensorflow as tf
import numpy as np


class FBODEEquation(object):
    def __init__(self, beta, gamma, kappa, lambda1, lambda2, lambda3, cost_I, cost_lambda1, g, Delta_t):
        # parameters of the MFG
        self.beta = beta
        self.gamma = gamma
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.cost_I = cost_I # penalty for being in I
        self.cost_lambda1 = cost_lambda1
        self.kappa = kappa
        self.g = 0.2
        self.Delta_t = Delta_t

    def driver_P(self, Z_empirical, P, ALPHA, beta_vector):
        n_samples =    tf.shape(P)[0]
        Nstates =      tf.shape(P)[1]
        driver =       np.zeros((n_samples, Nstates))
        driver[:,0] =  tf.reshape(-beta_vector * ALPHA * Z_empirical * tf.reshape(P[:,0],[n_samples,1]) + self.kappa * tf.reshape(P[:,2],[n_samples,1]), [n_samples])
        driver[:,1] =  tf.reshape(beta_vector * ALPHA * Z_empirical * tf.reshape(P[:,0],[n_samples,1]) - self.gamma * tf.reshape(P[:,1],[n_samples,1]), [n_samples])
        driver[:,2] =  tf.reshape(self.gamma * tf.reshape(P[:,1],[n_samples,1]) - self.kappa * tf.reshape(P[:,2],[n_samples,1]), [n_samples])
        return driver


    def driver_U(self, Z_empirical, U, ALPHA, beta_vector):
        n_samples =    tf.shape(U)[0]
        Nstates =      tf.shape(U)[1]
        driver =       np.zeros((n_samples, Nstates))
        driver[:,0] =   tf.reshape(beta_vector* ALPHA * Z_empirical * (tf.reshape(U[:,0],[n_samples,1])-tf.reshape(U[:,1],[n_samples,1])) - \
                       0.5 * self.cost_lambda1 * ((self.lambda1-ALPHA)**2), [n_samples])
        driver[:,1] =   tf.reshape(self.gamma * (tf.reshape(U[:,1],[n_samples,1])-tf.reshape(U[:,2],[n_samples,1])) - self.cost_I, [n_samples])
        driver[:,2] =   tf.reshape(self.kappa * (tf.reshape(U[:,2],[n_samples,1])-tf.reshape(U[:,0],[n_samples,1])), [n_samples])
        return driver



    def optimal_ALPHA(self, U, Z_empirical, beta_vector):
        n_samples = tf.shape(U)[0]
        alpha =     self.lambda1 + (beta_vector/self.cost_lambda1) * Z_empirical * (tf.reshape(U[:,0],[n_samples,1])-tf.reshape(U[:,1],[n_samples,1]))
        return alpha