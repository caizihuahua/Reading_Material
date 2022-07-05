import time,os
import numpy as np
import tensorflow as tf
from pinn import PINN
from prp_dat import *
from make_fig import *

def main():
    tmin, tmax =  0., 1.
    xmin, xmax = -1., 1.
    
    N_nabla_rho = 640
    N_p=50
    x_star = 0.0
    N_f = 2000

    epoch = 8000
    batch = 2**8
    tol = 1e-6

    lr0 = 4e-3
    gam = 1e-2
    lrd_cos = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate = lr0, 
        decay_steps = epoch, 
        alpha = gam
        )
    lr = lrd_cos

    t_nabla_rho, x_nabla_rho, \
    t_p, x_p, t_f, x_f = prp_dataset(tmin, tmax, xmin, xmax,
                                    N_nabla_rho,
                                    N_p, x_star,
                                    N_f)
    u_p = func_up(t_p,x_p)
    
    pinn = PINN(t_nabla_rho, x_nabla_rho,
                t_p, x_p,u_p,
                t_f, x_f,
                in_dim = 2, out_dim=3, width=20, depth=7, activ = "tanh",
                w_init = "glorot_normal", b_init = "zeros", 
                lr = lr, opt = "Adam")

    # if os.path.exists('./pro_saved_model'):
    #     # 只保留模型dnn，pinn中的其他都未保存
    #     print('load the previous model')
    #     pinn.dnn = tf.saved_model.load('./pro_saved_model')
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
    u_hat = pinn.predict(t, x)
    t1 = time.time()
    elps = t1 - t0
    print("elapsed time for PINN inference (sec):", elps)
    print("elapsed time for PINN inference (min):", elps / 60.)
    plot_sol1(TX, u_hat[:,0].numpy(),title='density_prediction')
    plot_sol1(TX, u_hat[:,1].numpy(),title='velocity_prediction')
    plot_sol1(TX, u_hat[:,2].numpy(),title='pressure_prediction')

    # pinn.dnn.save("pro_saved_model")

if __name__ == "__main__":
    main()
