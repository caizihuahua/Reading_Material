import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os,time,datetime,sys

class PINN(tf.keras.Model):
    def __init__(
            self,
            t,x,u,t_r,x_r,lb,ub,
            in_dim,out_dim,width,depth,
            activ="tanh",w_init="glorot_normal",b_init="zeros",
            lr=1e-3,opt="Adam",
            info_seed=1234):
        super().__init__()
        # information
        self.info_seed = info_seed
        # initial the data
        self.data_type = tf.float32
        self.x = tf.convert_to_tensor(x,dtype=self.data_type)
        self.t = tf.convert_to_tensor(t,dtype=self.data_type)
        self.u = tf.convert_to_tensor(u,dtype=self.data_type)
        # pde loss train point = inner + outer(initial+boundary)
        t_r = tf.concat([t,t_r],axis=0)
        x_r = tf.concat([x,x_r],axis=0)
        self.x_r = tf.convert_to_tensor(x_r,dtype=self.data_type)
        self.t_r = tf.convert_to_tensor(t_r,dtype=self.data_type)
        self.lb = tf.convert_to_tensor(lb,dtype=self.data_type)
        self.ub = tf.convert_to_tensor(ub,dtype=self.data_type)
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

    def loss_icbc(self):
        u_nn = self.dnn(tf.concat([self.t,self.x],1))
        loss = tf.reduce_mean(tf.square(self.u-u_nn))
        return loss

    @tf.function
    def grad_desc(self):
        with tf.GradientTape() as tp:
            loss = self.loss_pde() + self.loss_icbc()
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

epoch = 12000
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


print("python    :", sys.version)
print("tensorflow:", tf.__version__)
print("rand seed :", info_seed)
os.environ["PYTHONHASHSEED"] = str(info_seed)

tmin, tmax = 0., 1.
xmin, xmax = 0., 1.
lb = tf.constant([tmin, xmin], dtype = tf.float32)
ub = tf.constant([tmax, xmax], dtype = tf.float32)

t_0 = tf.ones((N_0, 1), dtype = tf.float32) * lb[0]
x_0 = tf.random.uniform((N_0, 1), lb[1], ub[1], dtype = tf.float32)
u_0 = tf.convert_to_tensor([[1.4 if x>0.5 else 1.0 ,0.1,1.0] for x in x_0.numpy()],dtype=tf.float32)

t_b = tf.random.uniform((N_b, 1), lb[0], ub[0], dtype = tf.float32)
x_b = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((N_b, 1), .5, dtype = tf.float32)
u_b = tf.convert_to_tensor([[1.4 if x_b[i]<0.5+0.1*t_b[i] else 1.0 ,0.1 ,1.0] for i in range(len(x_b.numpy()))],dtype=tf.float32)

t = tf.concat([t_0,t_b],axis=0)
x = tf.concat([t_0,t_b],axis=0)
u = tf.concat([u_0,u_b],axis=0)

t_r_rand = tf.random.uniform((N_r, 1), lb[0], ub[0], dtype = tf.float32)
x_r_rand = tf.random.uniform((N_r, 1), lb[1], ub[1], dtype = tf.float32)

t_r_clus = tf.random.uniform((N_r, 1), lb[0], ub[0], dtype = tf.float32)
noise = 0.1
a = 0.1
b = 0.5
aa = a*(1+np.random.randn()*noise)
bb = b*(1+np.random.randn()*noise)
cc = np.random.choice(range(t_r_clus.shape[0]),size=int(N_r*0.6), replace=False)
x_r_clus =tf.convert_to_tensor([ [a*(1+np.random.randn()*noise)*t_r_clus[i].numpy()[0]+b*(1+np.random.randn()*noise)] if (i in cc) else [np.random.random()] for i in range(len(t_r_clus))],dtype=tf.float32)

pinn_clus = PINN(
            t,x,u,
            t_r_clus,x_r_clus,
            lb,ub,
            in_dim,out_dim,width,depth,
            act,w_init,b_init,
            lr,opt,info_seed)

pinn_clus.train(epoch,tol,info_freq)

pinn_rand = PINN(t,x,u,t_r_rand,x_r_rand,lb,ub,
            in_dim,out_dim,width,depth,
            act,w_init,b_init,
            lr,opt,info_seed)

pinn_rand.train(epoch,tol,info_freq)

t = np.linspace(tmin,tmax,1001)
x = np.linspace(xmin,xmax,101)
t,x = np.meshgrid(t,x)
t = t.reshape(-1, 1)
x = x.reshape(-1, 1)
TX = np.c_[t,x]

u_clus,r_clus = pinn_clus.predict(t,x)
u_rand,r_rand = pinn_rand.predict(t,x)
u = [u_clus,u_rand]

X = TX
lb = X.min(0)
ub = X.max(0)
x = np.linspace(lb[0],ub[0],200)
y = np.linspace(lb[1],ub[1],200)
x,y = np.meshgrid(x,y)

def plot_solution(X,u,savepath="./pics"):
    lb = X.min(0)
    ub = X.max(0)
    x = np.linspace(lb[0],ub[0],200)
    y = np.linspace(lb[1],ub[1],200)
    x,y = np.meshgrid(x,y)
    density_clus = griddata(X,u[0][:,0].numpy().flatten(),(x,y),method="linear")
    density_rand = griddata(X,u[0][:,0].numpy().flatten(),(x,y),method="linear")
    velocity_clus = griddata(X,u[0][:,1].numpy().flatten(),(x,y),method="linear")
    velocity_rand = griddata(X,u[1][:,1].numpy().flatten(),(x,y),method="linear")
    pressure_clus = griddata(X,u[0][:,2].numpy().flatten(),(x,y),method="linear")
    pressure_rand = griddata(X,u[1][:,2].numpy().flatten(),(x,y),method="linear")

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fig1,ax1 =plt.subplots(1,2)
    img1 = plt.imshow(density_clus,interpolation='nearest',cmap='rainbow',extent=[0,1,0,1],origin="lower",aspect="auto")
    img2 = plt.imshow(density_rand,interpolation='nearest',cmap='rainbow',extent=[0,1,0,1],origin="lower",aspect="auto")
    plt.colorbar(img1,ax=ax1[0])
    plt.colorbar(img2,ax=ax1[1])
    ax1[0].title("Clus density")
    ax1[0].xlabel('t')
    ax1[0].ylabel('x')
    plt.savefig(savepath+"/clus density")

    fig2,ax2 =plt.subplots(1,2)
    img1 = plt.imshow(velocity_clus,interpolation='nearest',cmap='rainbow',extent=[0,1,0,1],origin="lower",aspect="auto")
    img2 = plt.imshow(velocity_rand,interpolation='nearest',cmap='rainbow',extent=[0,1,0,1],origin="lower",aspect="auto")
    plt.colorbar(img1,ax=ax2[0])
    plt.colorbar(img2,ax=ax2[1])
    ax2[0].title("Clus velocity")
    ax2[0].xlabel('t')
    ax2[0].ylabel('x')
    plt.savefig(savepath+"/clus velocity")

    fig3,ax3 =plt.subplots(1,2)
    img1 = plt.imshow(pressure_clus,interpolation='nearest',cmap='rainbow',extent=[0,1,0,1],origin="lower",aspect="auto")
    img2 = plt.imshow(pressure_rand,interpolation='nearest',cmap='rainbow',extent=[0,1,0,1],origin="lower",aspect="auto")
    plt.colorbar(img1,ax=ax3[0])
    plt.colorbar(img2,ax=ax3[1])
    ax3[0].title("Clus pressure")
    ax3[0].xlabel('t')
    ax3[0].ylabel('x')
    plt.savefig(savepath+"/clus pressure")
    

plot_solution(TX,u)