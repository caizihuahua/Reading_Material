import time,os
import numpy as np
import tensorflow as tf

from pinn import PINN
from prp_dat import func_u0, func_ub, prp_grd, prp_dataset
from make_fig import *

def main():
    tmin, tmax =  0., 1.
    xmin, xmax = -1., 1.
    N_0 = 60
    N_b = 60
    N_r = 1500

    epoch = 5000
    batch = 2**8
    tol = 1e-5

    lr0 = 1e-3
    gam = 1e-2
    lrd_cos = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate = lr0, 
        decay_steps = epoch, 
        alpha = gam
        )
    lr = lrd_cos

    t_0, x_0, t_b, x_b, t_r, x_r = prp_dataset(tmin, tmax, xmin, xmax, N_0, N_b, N_r)
    u_0 = func_u0(x_0)
    u_b = func_ub(x_b)

    pinn = PINN(t_0, x_0, u_0, 
                t_b, x_b, u_b, 
                t_r, x_r, 
                in_dim = 2, out_dim=1, width=20, depth=8, activ = "tanh",
                w_init = "glorot_normal", b_init = "zeros", 
                lr = lr, opt = "Adam")

    # if os.path.exists('./burgers_saved_model'):
    #     # 只保留模型dnn，pinn中的其他都未保存
    #     print('load the previous model')
    #     pinn.dnn = tf.saved_model.load('./burgers_saved_model')
    # else:
    pinn.train(epoch, batch, tol)
    plot_loss_log(pinn.ep_log,pinn.loss_log,'train_loss_log')
    plot_loss(pinn.ep_log,pinn.loss_log,'train_loss')
        
    # PINN inference
    nt = int(1e3) + 1
    nx = int(1e2) + 1
    t, x, TX = prp_grd(
        tmin, tmax, nt, 
        xmin, xmax, nx
    )
    t0 = time.time()
    u_hat, f = pinn.predict(t, x)
    t1 = time.time()
    elps = t1 - t0
    print("elapsed time for PINN inference (sec):", elps)
    print("elapsed time for PINN inference (min):", elps / 60.)
    plot_sol1(TX, u_hat.numpy(),title='prediction')
    plot_sol1(TX, f.numpy(),title='Euler equation loss')

    # pinn.dnn.save("burgers_saved_model")

if __name__ == "__main__":
    main()
