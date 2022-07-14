import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os,time,datetime,sys

class PINN(tf.keras.Model):
    def __init__(
            self,
            t_i,x_i,u_i,t_b,x_b,t_r,x_r,lb,ub,
            in_dim,out_dim,width,depth,
            activ="tanh",w_init="glorot_normal",b_init="zeros",
            lr=1e-3,opt="Adam",weight_data=1.,weight_pde=1.,
            info_seed=1234):
        super().__init__()
        # information
        self.info_seed = info_seed
        # initial the data
        self.data_type = tf.float32
        self.x_i = tf.convert_to_tensor(x_i,dtype=self.data_type)
        self.t_i = tf.convert_to_tensor(t_i,dtype=self.data_type)
        self.u_i = tf.convert_to_tensor(u_i,dtype=self.data_type)
        # self.u_i = tf.cast(u_i,dtype=self.data_type)
        self.x_b = tf.convert_to_tensor(x_b,dtype=self.data_type)
        self.t_b = tf.convert_to_tensor(t_b,dtype=self.data_type)
        # pde loss train point = inner + outer(initial+boundary)
        t_r = tf.concat([t_i,t_b,t_r],axis=0)
        x_r = tf.concat([x_i,x_b,x_r],axis=0)
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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)

        # track loss
        self.ep_log = []
        self.loss_log = []

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

    
    def loss_pde(self):
        with tf.GradientTape(persistent=True) as tp:
            tp.watch(self.t_r)
            tp.watch(self.x_r)
            u = self.dnn(tf.concat([self.t_r,self.x_r],1))
            rho = u[:,0][:,None]
            v = u[:,1][:,None]
            p = u[:,2][:,None]
        rho_t = tp.gradient(rho,self.t_r)
        v_t = tp.gradient(v,self.t_r)
        rho_x = tp.gradient(rho,self.x_r)
        v_x = tp.gradient(v,self.x_r)
        p_x = tp.gradient(p,self.x_r)
        del tp
        equ_1 = rho_t + rho_x*v + rho*v_x
        equ_2 = (rho_t*v + rho*v_t) + (rho*(2*v*v_x) +(v**2)*rho_x + p_x)
        r = tf.reduce_mean(tf.square(equ_1)+tf.square(equ_2))
        return r

    def loss_ic(self):
        u_nn_init = self.dnn(tf.concat([self.t_i,self.x_i],1))
        loss_init = tf.reduce_mean(tf.square(self.u_i-u_nn_init))
        return loss_init
    def loss_bc(self):
        # boundary xmin
        xmin_index = [i for i,x in enumerate(self.x_b.numpy()) if x==self.lb[1]]
        xmin_t = tf.convert_to_tensor([self.t_b[i] for i in xmin_index],dtype=self.data_type)
        xmin_x = tf.convert_to_tensor([self.x_b[i] for i in xmin_index],dtype=self.data_type)
        xmin_u = tf.convert_to_tensor([self.u_b[i] for i in xmin_index],dtype=self.data_type)
        # boundary xmax
        xmin_index = [i for i,x in enumerate(self.x_b.numpy()) if x==self.ub[1]]
        xmax_t = tf.convert_to_tensor([self.t_b[i] for i in xmax_index],dtype=self.data_type)
        xmax_x = tf.convert_to_tensor([self.x_b[i] for i in xmax_index],dtype=self.data_type)
        xmax_u = tf.convert_to_tensor([self.u_b[i] for i in xmax_index],dtype=self.data_type)
        
        with tf.GradientTape(persistent=True) as tp:
            tp.watch(xmin_x)
            tp.watch(xmax_x)
            u_nn_xmin = self.dnn(tf.concat([xmin_t,xmin_x],1))
            u_nn_xmax = self.dnn(tf.concat([xmax_t,xmax_x],1))
        tmp1 = tp.gradient(u_nn_xmax,xmax)
        tmp2 = tp.gradient(u_nn_xmin,xmin)
        loss_b1 = tf.reduce_mean(tf.square(u_nn_xmax-u_nn_xmin))
        loss_b2 = tf.reduce_mean(tf.square(tmp1-tmp2))
        return loss_b1+loss_b2
    
    @tf.function
    def grad_desc(self):
        with tf.GradientTape() as tp:
            loss = self.loss_pde() + self.loss_ic() + self.loss_ic()
        grad = tp.gradient(loss,self.params)
        del tp
        self.optimizer.apply_gradients(zip(grad,self.params))
        return loss
    
    def train(self,epoch,tol,info_freq):
        print(">>>>> training setting;")
        print("         # of epoch     :", epoch)
        print("         convergence tol:", tol)
        t0 = time.time()
        for ep in range(epoch+1):
            ep_loss = self.grad_desc()
            if ep % info_freq ==0:
                elps = time.time() -t0
                self.ep_log.append(ep)
                self.loss_log.append(ep_loss)
                print("ep: %d, loss: %.3e, elps: %.3f" % (ep, ep_loss, elps))
                t0 = time.time()
            if ep_loss < tol:
                print(">>>>> end time:", datetime.datetime.now())
                break
        print(">>>>> end time:", datetime.datetime.now())

    def predict(self,t,x):
        t = tf.convert_to_tensor(t,dtype=self.data_type)
        x = tf.convert_to_tensor(x,dtype=self.data_type)
        with tf.GradientTape(persistent=True) as tp:
            tp.watch(t)
            tp.watch(x)
            u = self.dnn(tf.concat([t,x],1))
            rho = u[:,0][:,None]
            v = u[:,1][:,None]
            p = u[:,2][:,None]
        rho_t = tp.gradient(rho,t)
        v_t = tp.gradient(v,t)
        rho_x = tp.gradient(rho,x)
        v_x = tp.gradient(v,x)
        p_x = tp.gradient(p,x)
        del tp
        equ_1 = rho_t + rho_x*v + rho*v_x
        equ_2 = (rho_t*v + rho*v_t) + (rho*(2*v*v_x) +(v**2)*rho_x + p_x)
        gv = tf.square(equ_1)+tf.square(equ_2)
        return u,gv

in_dim = 2
out_dim = 3
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
info_freq = 500
info_seed = 1234

weight_data = 1.
weight_pde = 1.

print("python    :", sys.version)
print("tensorflow:", tf.__version__)
print("rand seed :", info_seed)
os.environ["PYTHONHASHSEED"] = str(info_seed)

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

u_0 = tf.convert_to_tensor([[1.+0.2*np.sin(np.pi*a),1.,1.] for a in x_0.numpy().flatten()],dtype=tf.float32)

pinn = PINN(t_0,x_0,u_0,t_b,x_b,t_r,x_r,lb,ub,
            in_dim,out_dim,width,depth,
            act,w_init,b_init,
            lr,opt,weight_data,weight_pde,
            info_seed)

def plot_solution(X,u,savepath="./pics"):
    lb = X.min(0)
    ub = X.max(0)
    x = np.linspace(lb[0],ub[0],200)
    y = np.linspace(lb[1],ub[1],200)
    x,y = np.meshgrid(x,y)
    phi = griddata(X,u[:0].numpy().flatten().reshape(-1,1),(x,y),method="linear")
    plt.imshow(phi,interpolation='nearest',cmap='rainbow',extent=[0,1,-1,1],origin="lower",aspect="auto")
    plt.colorbar()
    plt.title("density prediction")
    plt.xlabel('t')
    plt.ylabel('x')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    plt.savefig(savepath+'/'+title)

t = np.linspace(tmin,tmax,1001)
x = np.linspace(xmin,xmax,101)

t,x = np.meshgrid(t,x)
t = t.reshape(-1, 1)
x = x.reshape(-1, 1)
TX = np.c_[t,x]
u_hat,r_hat = pinn.predict(t,x)
plot_solution(TX, u_hat)