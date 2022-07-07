import os
import time
import datetime
import tensorflow as tf
import numpy as np

class PINN(tf.keras.Model):
    def __init__(self, 
                 t_0, x_0, u_0, 
                 t_b, x_b, u_b, 
                 t_f, x_f, 
                 in_dim, out_dim, width, depth, activ = "tanh", 
                 w_init = "glorot_normal", b_init = "zeros", 
                 lr = 1e-3, opt = "Adam",
                 freq_info = 100, r_seed = 1234):
        # initialize the configuration
        super().__init__()
        self.r_seed = r_seed
        self.random_seed(seed = r_seed)
        self.data_type  = tf.float32
        self.in_dim     = in_dim       # input dimension
        self.out_dim     = out_dim       # output dimension
        self.width     = width       # internal dimension
        self.depth  = depth    # (# of hidden layers) + output layer
        self.activ  = activ    # activation function
        self.w_init = w_init   # initial weight
        self.b_init = b_init   # initial bias
        self.lr     = lr       # learning rate
        self.opt    = opt      # name of your optimizer
        self.freq_info = freq_info    # monitoring frequency

        # input-output pair
        self.t_0 = t_0; self.x_0 = x_0; self.u_0 = u_0   # evaluates initial condition
        self.t_b = t_b; self.x_b = x_b; self.u_b = u_b   # evaluates boundary condition
        self.t_f = t_f; self.x_f = x_f                   # evaluates domain residual
        
        # bounds
        X_r     = tf.concat([t_f, x_f], 1)
        self.lb = tf.cast(tf.reduce_min(X_r, axis = 0), self.data_type)
        self.ub = tf.cast(tf.reduce_max(X_r, axis = 0), self.data_type)
        
        # call
        self.dnn = self.dnn_init(in_dim, out_dim, width, depth)
        self.params = self.dnn.trainable_variables
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
        
        # parameter setting
        self.nu = tf.constant(.01 / np.pi, dtype = self.data_type)

        # track loss
        self.ep_log = []
        self.loss_log = []
        
        print("\n************************************************************")
        print("****************     MAIN PROGRAM START     ****************")
        print("************************************************************")
        print(">>>>> start time:", datetime.datetime.now())
        print(">>>>> configuration;")
        print("         dtype        :", self.data_type)
        print("         activ func   :", self.activ)
        print("         weight init  :", self.w_init)
        print("         learning rate:", self.lr)
        print("         optimizer    :", self.opt)
        print("         summary      :", self.dnn.summary())
        
    def random_seed(self, seed = 1234):
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
    def dnn_init(self, in_dim, out_dim, width, depth):
        # network configuration (N: in_dim -> out_dim (in_dim -> width -> ... -> width -> out_dim))
        network = tf.keras.Sequential()
        network.add(tf.keras.layers.InputLayer(in_dim))
        network.add(tf.keras.layers.Lambda(lambda x: 2. * (x - self.lb) / (self.ub - self.lb) - 1.))
        # construct the network
        for l in range(depth - 1):
            network.add(tf.keras.layers.Dense(width, activation = self.activ, use_bias = True,
                                                kernel_initializer = self.w_init, bias_initializer = self.b_init, 
                                                kernel_regularizer = None, bias_regularizer = None, 
                                                activity_regularizer = None, kernel_constraint = None, bias_constraint = None))
        network.add(tf.keras.layers.Dense(out_dim))
        return network
    
    def PDE(self, t, x):
        t = tf.convert_to_tensor(t, dtype = self.data_type)
        x = tf.convert_to_tensor(x, dtype = self.data_type)
        with tf.GradientTape(persistent = True) as tp:
            tp.watch(t)
            tp.watch(x)
            u = self.dnn(tf.concat([t, x], 1))
            u_x = tp.gradient(u, x)
        u_t  = tp.gradient(u, t)
        u_xx = tp.gradient(u_x, x)
        del tp
        f = u_t + u * u_x - self.nu * u_xx
        return u, f

    @tf.function
    def loss_glb(self, 
                 t_0, x_0, u_0, 
                 t_b, x_b, u_b, 
                 t_f, x_f):
        loss_0_u_hat,loss_0_f = self.PDE(t_0, x_0)
        loss_b_u_hat,loss_b_f = self.PDE(t_b, x_b)
        f = self.PDE(t_f,x_f)
        loss_0 = tf.reduce_mean(tf.square(u_0 - loss_0_u_hat))
        loss_b = tf.reduce_mean(tf.square(u_b - loss_b_u_hat))
        loss_f = tf.reduce_mean(tf.square(f))
        loss_glb = loss_0 + loss_b + loss_f
        return loss_glb

    def loss_grad(self, 
                  t_0, x_0, u_0, 
                  t_b, x_b, u_b, 
                  t_f, x_f): 
        with tf.GradientTape(persistent = True) as tp:
            loss = self.loss_glb(t_0, x_0, u_0, 
                                 t_b, x_b, u_b, 
                                 t_f, x_f)
        grad = tp.gradient(loss, self.params)
        del tp
        return loss, grad
    
    @tf.function
    def grad_desc(self, 
                  t_0, x_0, u_0, 
                  t_b, x_b, u_b, 
                  t_f, x_f):
        loss, grad = self.loss_grad(t_0, x_0, u_0, 
                                    t_b, x_b, u_b, 
                                    t_f, x_f)
        self.optimizer.apply_gradients(zip(grad, self.params))
        return loss
        
    def train(self, epoch = 10 ** 5, batch = 2 ** 6, tol = 1e-5): 
        print(">>>>> training setting;")
        print("         # of epoch     :", epoch)
        print("         batch size     :", batch)
        print("         convergence tol:", tol)
        
        t0 = time.time()
        
        # I had to convert input data (tf.tensor) into numpy style in order for mini-batch training (slicing)
        # and this works well for both full-batch and mini-batch training
        t_0 = self.t_0.numpy(); x_0 = self.x_0.numpy(); u_0 = self.u_0.numpy()
        t_b = self.t_b.numpy(); x_b = self.x_b.numpy(); u_b = self.u_b.numpy()
        t_f = self.t_f.numpy(); x_f = self.x_f.numpy()
        
        for ep in range(epoch):
            ep_loss = 0
            n_r = self.x_f.shape[0]
            idx_f = np.random.permutation(n_r)
            for idx in range(0, n_r, batch):
                # batch for domain residual
                t_f_btch = tf.convert_to_tensor(t_f[idx_f[idx: idx + batch if idx + batch < n_r else n_r]], dtype = self.data_type)
                x_f_btch = tf.convert_to_tensor(x_f[idx_f[idx: idx + batch if idx + batch < n_r else n_r]], dtype = self.data_type)
                # compute loss and perform gradient descent
                loss_btch = self.grad_desc(t_0, x_0, u_0, 
                                            t_b, x_b, u_b, 
                                            t_f_btch, x_f_btch)
                ep_loss += loss_btch / int(n_r / batch)
                
            if ep % self.freq_info == 0:
                elps = time.time() - t0
                self.ep_log.append(ep)
                self.loss_log.append(ep_loss)
                print("ep: %d, loss: %.3e, elps: %.3f" % (ep, ep_loss, elps))
                t0 = time.time()
            
            if ep_loss < tol:
                print(">>>>> program terminating with the loss converging to its tolerance.")
                print("\n************************************************************")
                print("*****************     MAIN PROGRAM END     *****************")
                print("************************************************************")
                print(">>>>> end time:", datetime.datetime.now())
                break
        
        print("\n************************************************************")
        print("*****************     MAIN PROGRAM END     *****************")
        print("************************************************************")
        print(">>>>> end time:", datetime.datetime.now())
                
    def predict(self, t, x):
        u_hat, f = self.PDE(t, x)
        return u_hat, f
