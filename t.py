import numpy as np
import matplotlib.pyplot as plt
import os
def plot_loss(x,y,title,savepath="./pics"):
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
    # plt.figure(figsize=(8,4))
    fig1,ax1 = plt.subplots(1)
    ax1.plot(x,y)
    ax1.grid(alpha=0.5)
    ax1.set_ylabel("loss")
    ax1.set_xlabel("epoch")
    ax1.set_title("train loss")

    # plt.figure(figsize=(8,4))
    fig2,ax2 = plt.subplots(1)
    ax2.plot(x,y)
    ax2.grid(alpha=0.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("loss")
    ax2.set_xlabel("epoch")
    ax1.set_title("train loss(log)")

    # plt.figure(figsize=(8,4))
    fig3,ax3 = plt.subplots(1)
    strt = int(len(x)*0.7)
    ax3.plot(x[strt:],y[strt:])
    # ax3.plot(x,y)
    ax3.grid(alpha=0.5)
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("loss")
    ax3.set_title("train loss(part)")

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    fig1.savefig(savepath+"/"+title)
    fig2.savefig(savepath+"/"+title+"(log)")
    fig3.savefig(savepath+"/"+title+"(part")

x = np.array([1,2,3,4,5,6,7])
y = x**2+4


plot_loss(x, y, "title")
