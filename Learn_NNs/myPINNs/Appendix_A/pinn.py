import os
import time
import datetime
import tensorflow as tf
import numpy as np

class PINN(tf.keras.Model):
    def __init__(self, 
                 t_0, x_0, u_0, 
                 t_b, x_b,
                 t_f, x_f, 
                 in_dim, out_dim, width, depth, activ = "tanh", 
                 w_init = "glorot_normal", b_init = "zeros", 
                 lr = 1e-3, opt = "Adam",
                 freq_info = 10, r_seed = 1234):
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
        self.t_b = t_b; self.x_b = x_b;                  # evaluates boundary condition
        self.t_f = t_f; self.x_f = x_f                   # evaluates domain residual
        
        # bounds
        X_r     = tf.concat([t_f, x_f], 1)
        self.lb = tf.cast(tf.reduce_min(X_r, axis = 0), self.data_type)
        self.ub = tf.cast(tf.reduce_max(X_r, axis = 0), self.data_type)
        
        # call
        self.dnn = self.dnn_init(in_dim, out_dim, width, depth)
        self.params = self.dnn.trainable_variables
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)
        
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
    
    def loss_PDE(self, t, x):
        t = tf.convert_to_tensor(t, dtype = self.data_type)
        x = tf.convert_to_tensor(x, dtype = self.data_type)
        with tf.GradientTape(persistent = True) as tp:
            tp.watch(t)
            tp.watch(x)
            # u = [\rho, v, p]
            u = self.dnn(tf.concat([t, x], 1))
            rho = u[:,0][:,None] # tf.convert_to_tensor(u[:,0].numpy().reshape(-1,1),dtype=tf.float32)
            v = u[:,1][:,None] #tf.convert_to_tensor(u[:,1].numpy().reshape(-1,1),dtype=tf.float32)
            p = u[:,2][:,None] #tf.convert_to_tensor(u[:,2].numpy().reshape(-1,1),dtype=tf.float32)
        rho_t = tp.gradient(rho,t)
        v_t = tp.gradient(v,t)
        rho_x = tp.gradient(rho,x)
        v_x = tp.gradient(v,x)
        p_x = tp.gradient(p,x)
        equ_1 = rho_t + rho_x*v + rho*v_x
        equ_2 = (rho_t*v + rho*v_t) + (rho*(2*v*v_x) +(v**2)*rho_x + p_x)
        del tp
        loss_f = tf.reduce_mean(tf.square(equ_1)+tf.square(equ_2))
        return loss_f

    def loss_bounday(self,t,x_u,x_l):
        t = tf.convert_to_tensor(t, dtype = self.data_type)
        x_u = tf.convert_to_tensor(x_u, dtype = self.data_type)
        x_l = tf.convert_to_tensor(x_l, dtype = self.data_type)
        with tf.GradientTape(persistent = True) as tp:
            tp.watch(t)
            tp.watch(x_u)
            tp.watch(x_l)
            u_u = self.dnn(tf.concat([t, x_u], 1))
            u_l = self.dnn(tf.concat([t, x_l], 1))
            rho_u = u_u[:,0]
            v_u = u_u[:,1]
            p_u = u_u[:,2]
            rho_l = u_l[:,0]
            v_l = u_l[:,1]
            p_l = u_l[:,2]
        rho_x_u = tp.gradient(rho_u,x_u)
        v_x_u = tp.gradient(v_u,x_u)
        p_x_u = tp.gradient(p_u,x_u)
        rho_x_l = tp.gradient(rho_l,x_l)
        v_x_l = tp.gradient(v_l,x_l)
        p_x_l = tp.gradient(p_l,x_l)
        nabla_u_loss = tf.square(rho_x_u-rho_x_l)+tf.square(v_x_u-v_x_l)+tf.square(p_x_u-p_x_l)
        u_loss = tf.square(u_u[:,0][:,None]-u_l[:,0][:,None])+tf.square(u_u[:,1][:,None]-u_l[:,1][:,None])+tf.square(u_u[:,2][:,None]-u_l[:,2][:,None])
        del tp
        return tf.reduce_mean(nabla_u_loss) +tf.reduce_mean(u_loss)
        
    def loss_init(self,t,x,u_0):
        t = tf.convert_to_tensor(t, dtype = self.data_type)
        x = tf.convert_to_tensor(x, dtype = self.data_type)
        u_0 = tf.convert_to_tensor(u_0,dtype = self.data_type)
        u = self.dnn(tf.concat([t, x], 1))
        return tf.reduce_mean(tf.reduce_sum(tf.square(u-u_0),1))
    
    @tf.function
    def grad_desc(self, 
                  t_0, x_0, u_0, 
                  t_b, x_b,
                  t_f, x_f):
        with tf.GradientTape(persistent = True) as tp:
            loss = self.loss_PDE(t_f,x_f)+self.loss_bounday(t_b,x_b[0],x_b[1])+self.loss_init(t_0,x_0,u_0)
        grad = tp.gradient(loss, self.params)
        del tp
        self.optimizer.apply_gradients(zip(grad, self.params))
        return loss
        
    def train(self, epoch = 10 ** 5, batch = 2 ** 6, tol = 1e-5): 
        print(">>>>> training setting;")
        print("         # of epoch     :", epoch)
        print("         batch size     :", batch)
        print("         convergence tol:", tol)
        t0 = time.time()
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
                loss_btch = self.grad_desc(self.t_0, self.x_0, self.u_0, self.t_b, self.x_b,t_f_btch, x_f_btch)
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
        t = tf.convert_to_tensor(t, dtype = self.data_type)
        x = tf.convert_to_tensor(x, dtype = self.data_type)
        return self.dnn(tf.concat([t, x], 1))
