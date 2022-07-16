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
        self.dnn = self.dnn_init()
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
    
    def dnn_init(self):
        net = tf.keras.Sequential()
        net.add(tf.keras.layers.InputLayer(self.in_dim))
        # net.add(tf.keras.layers.Lambda(lambda x: 2. * (x - self.lb) / (self.ub - self.lb) - 1.))
        for l in range(self.depth - 1):
            net.add(tf.keras.layers.Dense(units=self.width, activation = self.activ,kernel_initializer = self.w_init, bias_initializer = self.b_init))
        net.add(tf.keras.layers.Dense(self.out_dim))
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
        return tf.reduce_mean(tf.square(self.u-u_nn))*3

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
            self.loss_pde()
            self.loss_icbc()
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
        return self.dnn(tf.concat([t,x],1))

in_dim = 2
out_dim = 3
width = 20
depth = 7

epoch = 2000
tol = 1e-8

N_0 = 60
N_b = 60
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

tmin, tmax = 0., 2.
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

def plot_loss(pinn,savepath="./pics"):
    title = "loss"
    # plt.figure(figsize=(8,4))
    fig1,ax1 = plt.subplots(1)
    ax1.plot(pinn[0].ep_log,pinn[0].loss_log,label="Clus")
    ax1.plot(pinn[1].ep_log,pinn[1].loss_log,label="Rand")
    ax1.grid(alpha=0.5)
    ax1.set_ylabel("loss")
    ax1.set_xlabel("epoch")
    ax1.set_title("train loss")
    ax1.legend(loc='upper right')

    # plt.figure(figsize=(8,4))
    fig2,ax2 = plt.subplots(1)
    ax2.plot(pinn[0].ep_log,pinn[0].loss_log,label="Clus")
    ax2.plot(pinn[1].ep_log,pinn[1].loss_log,label="Rand")
    ax2.grid(alpha=0.5)
    ax2.set_ylabel("loss")
    ax2.set_xlabel("epoch")
    ax2.set_yscale("log")
    ax2.set_title("train loss(log)")
    ax2.legend(loc='upper right')

    # plt.figure(figsize=(8,4))
    fig3,ax3 = plt.subplots(1)
    strt = int(len(pinn[0].ep_log)*0.7)
    ax3.plot(pinn[0].ep_log[strt:],pinn[0].loss_log[strt:],label="Clus")
    ax3.plot(pinn[1].ep_log[strt:],pinn[1].loss_log[strt:],label="Rand")
    ax3.grid(alpha=0.5)
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("loss")
    ax3.legend(loc='upper right')
    ax3.set_title("train loss(part)")

    # if not os.path.exists(savepath):
    #    os.makedirs(savepath)
    # fig1.savefig(savepath+"/"+title)
    # fig2.savefig(savepath+"/"+title+"(log)")
    # fig3.savefig(savepath+"/"+title+"(part")


pinn = [pinn_clus,pinn_rand]
plot_loss(pinn)

u = [[1.4] if i[1]<0.5+0.1*i[0] else [1.0] for i in TX]
lb = TX.min(0)
ub = TX.max(0)
t = np.linspace(tmin,tmax,101)
x = np.linspace(xmin,xmax,101)
x,t = np.meshgrid(x,t)
density = griddata(TX,u,(t,x),method="linear")
fig,ax1 = plt.subplots(1)
img1 = ax1.imshow(density,interpolation='nearest',cmap='rainbow',extent=[0,1,0,2],origin="lower",aspect="auto")
plt.colorbar(img1,ax=ax1)

def plot_solution(X,u,savepath="./pics"):
    lb = X.min(0)
    ub = X.max(0)
    t = np.linspace(lb[0],ub[0],200)
    x = np.linspace(lb[1],ub[1],200)
    x,t = np.meshgrid(x,t)
    density_clus = griddata(X,u[0][:,0].numpy().flatten(),(t,x),method="linear")
    density_rand = griddata(X,u[0][:,0].numpy().flatten(),(t,x),method="linear")
    velocity_clus = griddata(X,u[0][:,1].numpy().flatten(),(t,x),method="linear")
    velocity_rand = griddata(X,u[1][:,1].numpy().flatten(),(t,x),method="linear")
    pressure_clus = griddata(X,u[0][:,2].numpy().flatten(),(t,x),method="linear")
    pressure_rand = griddata(X,u[1][:,2].numpy().flatten(),(t,x),method="linear")

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fig1,ax1 =plt.subplots(1,2)
    img1 = ax1[0].imshow(density_clus,interpolation='nearest',cmap='rainbow',extent=[0,1,0,1],origin="lower",aspect="auto")
    img2 = ax1[1].imshow(density_rand,interpolation='nearest',cmap='rainbow',extent=[0,1,0,1],origin="lower",aspect="auto")
    plt.colorbar(img1,ax=ax1[0])
    plt.colorbar(img2,ax=ax1[1])
    ax1[0].set_title("Clus density")
    ax1[1].set_title("Rand density")
    ax1[0].set_xlabel('t') 
    ax1[1].set_xlabel('t') 
    ax1[0].set_ylabel('x')
    # fig1.savefig(savepath+"/clus density")

    fig2,ax2 =plt.subplots(1,2)
    img1 = ax2[0].imshow(velocity_clus,interpolation='nearest',cmap='rainbow',extent=[0,1,0,1],origin="lower",aspect="auto")
    img2 = ax2[1].imshow(velocity_rand,interpolation='nearest',cmap='rainbow',extent=[0,1,0,1],origin="lower",aspect="auto")
    plt.colorbar(img1,ax=ax2[0])
    plt.colorbar(img2,ax=ax2[1])
    ax2[0].set_title("Clus velocity")
    ax2[1].set_title("Rand velocity")
    ax2[0].set_xlabel('t')
    ax2[1].set_xlabel('t')
    ax2[0].set_ylabel('x')
    # fig2.savefig(savepath+"/clus velocity")

    fig3,ax3 =plt.subplots(1,2)
    img1 = ax3[0].imshow(pressure_clus,interpolation='nearest',cmap='rainbow',extent=[0,1,0,1],origin="lower",aspect="auto")
    img2 = ax3[1].imshow(pressure_rand,interpolation='nearest',cmap='rainbow',extent=[0,1,0,1],origin="lower",aspect="auto")
    plt.colorbar(img1,ax=ax3[0])
    plt.colorbar(img2,ax=ax3[1])
    ax3[0].set_title("Clus pressure")
    ax3[1].set_title("Rand pressure")
    ax3[0].set_xlabel('t') 
    ax3[1].set_xlabel('t') 
    ax3[0].set_ylabel('x')
    # fig3.savefig(savepath+"/clus pressure")

def plot_final(X,u,final_t,savepath="./pics"):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    index = np.array([ [i,x[1]] for i,x in enumerate(TX) if x[0]==final_t])
    density_clus = [u[0][:,0][int(i)] for i in index[:,0]]
    density_rand = [u[0][:,0][int(i)] for i in index[:,0]]
    velocity_clus = [u[1][:,1][int(i)] for i in index[:,0]]
    velocity_rand = [u[1][:,1][int(i)] for i in index[:,0]]
    pressure_clus = [u[1][:,2][int(i)] for i in index[:,0]]
    pressure_rand = [u[1][:,2][int(i)] for i in index[:,0]]
    xx = index[:,1]

    fig1,ax1 = plt.subplots(1)
    ax1.set_title(f"density(t={final_t})")
    ax1.set_xlabel('x')
    ax1.set_ylabel('density')
    ax1.plot(xx,density_clus,"*",label="Clus NN")
    ax1.plot(xx,density_rand,"o",color="orange",label="Rand NN",markersize=3)
    yy = np.array([1.4 if x<0.5+0.1*final_t else 1.0 for x in xx])
    ax1.plot(xx,yy,color="r",label='exact')
    ax1.legend(loc="upper right")
    # fig1.tight_layout()
    # fig1.savefig(f"{savepath}/density(t={final_t}).png")

    fig2,ax2 = plt.subplots(1)
    ax2.set_title(f"velocity(t={final_t})")
    ax2.set_xlabel('x')
    ax2.set_ylabel('velocity')
    ax2.plot(xx,velocity_clus,"*",label="Clus NN")
    ax2.plot(xx,velocity_rand,"o",color="orange",label="Rand NN",markersize=3)
    ax2.plot(xx,np.ones(len(xx)),color="r",label="exact")
    ax2.legend(loc="upper right")
    ax2.axis(ymin=0.95,ymax=1.05)
    # fig2.savefig(f"{savepath}/velocity(t={final_t}.png")

    fig3,ax3 = plt.subplots(1)
    ax3.set_title(f"pressure(t={final_t})")
    ax3.set_xlabel('x')
    ax3.set_ylabel('pressure')
    ax3.plot(xx,pressure_clus,"*",label="Clus NN")
    ax3.plot(xx,pressure_rand,"o",color="orange",label="Rand NN",markersize=3)
    ax3.plot(xx,np.ones(len(xx)),color="r",label="exact")
    ax3.legend(loc="upper right")
    ax3.axis(ymin=0.95,ymax=1.05)
    # fig3.savefig(f"{savepath}/pressure(t={final_t}).png")

t = np.linspace(tmin,tmax,1001)
x = np.linspace(xmin,xmax,101)
t,x = np.meshgrid(t,x)
t = t.reshape(-1, 1)
x = x.reshape(-1, 1)
TX = np.c_[t,x]


u_clus,r_clus = pinn_clus.predict(t,x)
u_rand,r_rand = pinn_rand.predict(t,x)
u = [u_clus,u_rand]

plot_final(TX,u,tmax)

plot_solution(TX,u)

