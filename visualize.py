from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import matplotlib.pyplot as plt

def draw_function_3dsurface_and_path(f, x_interval, y_interval, path_points=None) -> None:
    """
    Draws a surface of a function in 3D space

        Parameters:
            f (function) -> float: a mathematical function, accepts pair of values (x1, x2)
            x_interval ((int, int)): displayed interval on x axis
            y_interva ((int, int)): displayed interval on y axis
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

    fig = plt.figure("F(x1, x2)")
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', zorder=1)
    ax.plot(x1, x2, z, color='orange', linewidth=2, zorder=3)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1,x2)')

    plt.show()