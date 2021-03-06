"""
********************************************************************************
prep data
********************************************************************************
"""

import numpy as np
import tensorflow as tf

def func_u0(x):
    return [[1.+0.2*np.sin(np.pi*a),1.,1.] for a in x.numpy().flatten()]

def prp_grd(tmin, tmax, nt,
            xmin, xmax, nx):
    t = np.linspace(tmin, tmax, nt)
    x = np.linspace(xmin, xmax, nx)
    t, x = np.meshgrid(t, x)
    t, x = t.reshape(-1, 1), x.reshape(-1, 1)
    TX = np.c_[t, x]
    return t, x, TX

def prp_dataset(tmin, tmax, xmin, xmax, N_0, N_b, N_r):
    lb = tf.constant([tmin, xmin], dtype = tf.float32)
    ub = tf.constant([tmax, xmax], dtype = tf.float32)
    t_0 = tf.ones((N_0, 1), dtype = tf.float32) * lb[0]
    x_0 = tf.random.uniform((N_0, 1), lb[1], ub[1], dtype = tf.float32)
    t_b = tf.random.uniform((N_b, 1), lb[0], ub[0], dtype = tf.float32)
    x_b = [tf.ones((N_b, 1), dtype = tf.float32) * ub[1],tf.ones((N_b, 1), dtype = tf.float32) * lb[1]]
    t_r = tf.random.uniform((N_r, 1), lb[0], ub[0], dtype = tf.float32)
    x_r = tf.random.uniform((N_r, 1), lb[1], ub[1], dtype = tf.float32)

    return t_0, x_0, t_b, x_b, t_r, x_r

