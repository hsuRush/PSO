# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from functions import rosenbrock

ax = None

def plot3d_init():
    global ax
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    """
    def log_tick_formatter(val, pos=None):
        return "{:.2e}".format(10**val)
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    """

import matplotlib.ticker as mticker





def plot_func(func=rosenbrock):

    s = 0.25   # Try s=1, 0.25, 0.1, or 0.05
    X = np.arange(-10, 10.+s, s)   #Could use linspace instead if dividing
    Y = np.arange(-10, 10.+s, s)   #evenly instead of stepping...
        
    #Create the mesh grid(s) for all X/Y combos.
    X, Y = np.meshgrid(X, Y)

    #Rosenbrock function w/ two parameters using numpy Arrays
    Z = func(X,Y)

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)  #Try coolwarm vs jet

    #ax.plot(X,Y,Z)
    
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

def plot_3d_dot(xs,ys,zs, **args):
    ax.scatter(xs,ys,zs, **args)

def plot(x,y, **args):
    plt.plot(x,y, **args)

def plot_show(pause_time=.0, **args):
    while True:
        try:
            plt.show(**args)
            #plt.draw()
            plt.pause(pause_time)

        except UnicodeDecodeError:
            continue
        break

   

if __name__ == "__main__":
    plot_func()
    plot_show()