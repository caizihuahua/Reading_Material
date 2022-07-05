"""
********************************************************************************
prep data
********************************************************************************
"""

import numpy as np
import tensorflow as tf

def func_up(t,x):
    return tf.ones((len(t.numpy()),1), dtype = tf.float32)

def func_ub(x):
    n = x.shape[0]
    return tf.zeros((n, 1), dtype = tf.float32)

def prp_grd(tmin, tmax, nt,
            xmin, xmax, nx):
    t = np.linspace(tmin, tmax, nt)
    x = np.linspace(xmin, xmax, nx)
    t, x = np.meshgrid(t, x)
    t, x = t.reshape(-1, 1), x.reshape(-1, 1)
    TX = np.c_[t, x]
    return t, x, TX

def prp_dataset(tmin, tmax, xmin, xmax, N_nabla_rho,N_p,x_star,N_f):
    t_nabla_rho = tf.random.uniform((N_nabla_rho, 1), tmin, tmax, dtype = tf.float32)
    x_nabla_rho = tf.random.uniform((N_nabla_rho,1), xmin, xmax, dtype = tf.float32)
    t_p = tf.random.uniform((N_p, 1), tmin, tmax, dtype = tf.float32)
    x_p = tf.ones((N_p,1),dtype=tf.float32) * x_star
    t_f = tf.random.uniform((N_f, 1), tmin, tmax, dtype = tf.float32)
    x_f = tf.random.uniform((N_f,1), xmin, xmax, dtype = tf.float32)

    return t_nabla_rho, x_nabla_rho, t_p, x_p, t_f, x_f