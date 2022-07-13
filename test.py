import tensorflow as tf
import numpy as np
import matplotlib as plt
import os,time,datetime,sys

in_dim = 2
out_dim = 1
width = 20
depth = 7

epoch = 8000
tol = 1e-8

N_0 = 50
N_b = 50
N_r = 2000

w_init = "glorot_normal"
b_init = "zeros"
act = "tanh"

lr = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate = 5e-3,
    decay_steps = epoch,
    alpha = 1e-2
)

opt = "Adam"
info_freq = 100
info_seed = 1234

weight_data = 1.
weight_pde = 1.

print("python    :", sys.version)
print("tensorflow:", tf.__version__)
print("rand seed :", info_seed)
os.environ["PYTHONHASHSEED"] = str(info_seed)
np.random.seed(info_seed)
tf.random.set_seed(info_seed)

tmin, tmax =  0., 1.
xmin, xmax = -1., 1.
lb = tf.constant([tmin, xmin], dtype = tf.float32)
ub = tf.constant([tmax, xmax], dtype = tf.float32)

t_0 = tf.ones((N_0, 1), dtype = tf.float32) * lb[0]
x_0 = tf.random.uniform((N_0, 1), lb[1], ub[1], dtype = tf.float32)
t_b = tf.random.uniform((N_b, 1), lb[0], ub[0], dtype = tf.float32)
x_b = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((N_b, 1), .5, dtype = tf.float32)
t_r = tf.random.uniform((N_r, 1), lb[0], ub[0], dtype = tf.float32)
x_r = tf.random.uniform((N_r, 1), lb[1], ub[1], dtype = tf.float32)

# initial and boundary
u_0 = -tf.sin(np.pi*x_0)
u_b = tf.zeros((x_b.shape[0],1),dtype=tf.float32)

t = tf.concat([t_0,t_b],axis=0)
x = tf.concat([x_0,x_b],axis=0)
u = tf.concat([u_0,u_b],axis=0)

class PINN(tf.keras.Model):
    def __init__(
            self,
            t,x,u,t_r,x_r,lb,ub,
            in_dim,out_dim,width,depth,
            activ="tanh",w_init="glorot_normal",b_init="zeros",
            lr=1e-3,opt="Adam",weight_data=1.,weight_pde=1.,
            info_freq=100,info_seed=1234):
        super().__init__()
        # information
        self.info_freq = info_freq
        self.info_seed = info_freq
        # initial the data
        self.data_type = tf.float32
        self.x = tf.convert_to_tensor(x,dtype=self.data_type)
        self.t = tf.convert_to_tensor(t,dtype=self.data_type)
        self.u = tf.convert_to_tensor(u,dtype=self.data_type)
        self.x_r = tf.convert_to_tensor(x_r,dtype=self.data_type)
        self.t_r = tf.convert_to_tensor(t_r,dtype=self.data_type)
        self.lb = tf.convert_to_tensor(lb,dtype=self.data_type)
        self.ub = tf.convert_to_tensor(ub,dtype=self.data_type)
        self.nu = tf.constant(0.01/np.pi,dtype=self.data_type)
        # neuron network configuration
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.depth = depth
        self.activ = activ
        self.w_init = w_init
        self.b_init = b_init
        self.lr = lr
        self.opt = opt
        self.weight_data = weight_data
        self.weight_pde = weight_pde
        
        # call
        self.dnn = self.dnn_init(in_dim,out_dim,width,depth)
        self.params = self.dnn.trainable_variables
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)

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
    
    def dnn_init(self,in_dim,out_dim,width,depth):
        net = tf.keras.Sequential()
        net.add(tf.keras.layers.InputLayer(in_dim))
        # net.add(tf.keras.layers.Lambda(lambda x: 2. * (x - self.lb) / (self.ub - self.lb) - 1.))

        for l in range(depth - 1):
            net.add(tf.keras.layers.Dense(units=width, activation = self.activ,kernel_initializer = self.w_init, bias_initializer = self.b_init, ))
        net.add(tf.keras.layers.Dense(out_dim))
        return net
    
    def loss_pde_bk(self):
        # with tf.GradientTape(persistent=True) as tp
        with tf.GradientTape(persistent=True) as tp:
            with tf.GradientTape(persistent=True) as tp2:
                tp.watch(self.t_r)
                tp.watch(self.x_r)
                u = self.dnn(tf.concat([self.t_r,self.x_r],1))
            u_t = tp2.gradient(u,self.t_r)
            u_x = tp2.gradient(u,self.x_r)
        u_xx = tp.gradient(u_x,self.x_r)
        del tp,tp2
        gv = u_t + u * u_x - self.nu * u_xx
        r = tf.reduce_mean(tf.square(gv))
        return r
    def loss_pde(self):
        with tf.GradientTape(persistent=True) as tp:
            tp.watch(self.t_r)
            tp.watch(self.x_r)
            u = self.dnn(tf.concat([self.t_r,self.x_r],1))
            u_t = tp.gradient(u,self.t_r)
            u_x = tp.gradient(u,self.x_r)
        u_xx = tp.gradient(u_x,self.x_r)
        del tp
        gv = u_t + u * u_x - self.nu * u_xx
        r = tf.reduce_mean(tf.square(gv))
        return r

    def loss_icbc(self):
        u_nn = self.dnn(tf.concat([self.t,self.x],1))
        return tf.reduce_mean(tf.square(self.u-u_nn))
    
    @tf.function
    def grad_desc(self):
        with tf.GradientTape() as tp:
            loss = self.loss_pde() + self.loss_icbc()
        grad = tp.gradient(loss,self.params)
        del tp
        self.optimizer.apply_gradients(zip(grad,self.params))
        return loss
    
    def train(self,epoch,tol):
        print(">>>>> training setting;")
        print("         # of epoch     :", epoch)
        print("         convergence tol:", tol)

        t0 = time.time()
        for ep in range(epoch):
            self.loss_pde()
            self.loss_icbc()
            ep_loss = self.grad_desc()
            if ep % self.info_freq ==0:
                elps = time.time() -t0
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

    def predict(self,t,x):
        with tf.GradientTape() as tp:
            with tf.GradientTape() as tp2:
                tp.watch(t)
                tp.watch(x)
                u = self.dnn(tf.concat([t,x],1))
            u_t = tp2.gradient(u,t)
            u_x = tp2.gradient(u,x)
        u_xx = tp.gradient(u_x,x)
        del tp,tp2
        gv = u_t + u * u_x - self.nu * u_xx
        r = tf.reduce_mean(tf.square(gv))
        return u,r


pinn = PINN(t,x,u,t_r,x_r,lb,ub,
            in_dim,out_dim,width,depth,
            activ=act,w_init=w_init,b_init=b_init,
            lr=1e-3,opt=opt,weight_data=1.,weight_pde=1.,
            info_freq=100,info_seed=1234)

pinn.train(epoch,tol)