import numpy as np
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

    
def plot_sol1(X_star, phi1,title):
    lb = X_star.min(0); ub = X_star.max(0)
    x, y = np.linspace(lb[0], ub[0], 200), np.linspace(lb[1], ub[1], 150); x, y = np.meshgrid(x, y)
    PHI_I = griddata(X_star, phi1.flatten(), (x, y), method = "linear")
    plt.figure(figsize = (12, 4))
    plt.imshow(PHI_I, interpolation='nearest',cmap='rainbow', extent=[0,1,-1,1], origin='lower', aspect='auto')
    plt.colorbar()
    plt.title(f'{title}')
    plt.xlabel('t')
    plt.ylabel('x')
    if not os.path.exists('./pics'):
        os.makedirs('./pics')
    plt.savefig(f'./pics/{title}')

def plot_loss_log(ep_log, loss_log,save_name):
    plt.figure(figsize = (8, 4))
    plt.plot(ep_log, loss_log, alpha = .7, label = "loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.grid(alpha = .5)
    plt.legend(loc = "upper right")
    plt.title('train loss(log)')
    if not os.path.exists('./pics'):
        os.makedirs('./pics')
    plt.savefig(f'./pics/{save_name}')

def plot_loss(ep_log, loss_log,save_name):
    plt.figure(figsize=(8,4))
    fig,ax = plt.subplots(2,1)
    strt = int(len(ep_log)*0.6)
    ax[0].plot(ep_log, loss_log)
    ax[0].grid(alpha=.5)
    ax[0].set_ylabel('loss')
    ax[0].set_title('train loss')
    ax[1].plot(ep_log[strt:], loss_log[strt:])
    ax[1].grid(alpha=.5)
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    if not os.path.exists('./pics'):
        os.makedirs('./pics')
    fig.savefig(f"./pics/{save_name}")