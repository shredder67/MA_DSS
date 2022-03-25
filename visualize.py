from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import matplotlib.pyplot as plt

def visualize_func_and_path(f, x_interval, y_interval, path_points=None) -> None:
    """
    Plots a surface of a function in 3D space, gradient map and search path over it's surface

        Parameters:
            f (function) -> float: a mathematical function, accepts pair of values (x1, x2)
            x_interval ((int, int)): displayed interval on x axis
            y_interval ((int, int)): displayed interval on y axis
            path_points ([np.array,...]): array of (x1, x2) pairs representing solution search
    """
    # 3D surface
    x = np.linspace(x_interval[0], x_interval[1], (x_interval[1] - x_interval[0])*3)
    y = np.linspace(y_interval[0], y_interval[1], (y_interval[1] - y_interval[0])*3)
    X, Y = np.meshgrid(x,y)
    Z = f((X, Y))

    # Path points transformation
    path_points = list(map(lambda x_vals: (x_vals[0], x_vals[1], f(x_vals) + 0.2),  path_points)) # add z-coordinates to points
    x1, x2, z = zip(*path_points)


    fig = plt.figure(figsize=(12, 8))
    
    # First plot - 3D surface
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', zorder=1)
    ax1.plot(x1, x2, z, color='orange', linewidth=2, zorder=3)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('f(x1,x2)')

    # Second plot - gradient map
    ax2 = fig.add_subplot(1, 2, 2)
    contours = plt.contour(X, Y, Z, 15)
    ax2.clabel(contours, inline=True, fontsize=10)

    ax2.plot(x1, x2)
    ax2.plot(x1, x2, '*')

    ax2.set_xlabel('x1', fontsize=11)
    ax2.set_ylabel('x2', fontsize=11)
    plt.colorbar()

    plt.show()