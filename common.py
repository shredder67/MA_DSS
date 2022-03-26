import numpy as np

def f(x_vals):
    x1, x2 = x_vals
    return 10*x1*x1 + 3*x1*x2 + x2*x2 + 10*x2


def calc_grad(f, n, x_vals):
    grad = np.zeros(n)
    d_x = .00001
    for i in range(n):
        f_0 = f(x_vals)
        x_vals[i] += d_x
        grad[i] = (f(x_vals) - f_0)/d_x
        x_vals[i] -= d_x
    return grad


def is_pos_def(matrix, n):
    def matrix_minor(arr, i, j):
        return arr[:i + 1, :j + 1]

    # Все угловые миноры должны быть положительны
    for i in range(n):
        if np.linalg.det(matrix_minor(matrix, i, i)) < 0: return False

    return True


def calc_hessian(f, n, x_vals):
    d_x = .00001
    hessian = np.full((n, n), 0)
    for i in range(n):
        for j in range(n):
            if i == j:
                x_vals[i] += d_x
                u1 = f(x_vals)
                x_vals[i] -= d_x
                u2 = f(x_vals)
                x_vals[i] -= d_x
                u3 = f(x_vals)
                x_vals[i] += d_x

                hessian[i][j] = (u1 - 2*u2 + u3)/d_x**2
            else:
                u1 = f(x_vals)
                x_vals[i] -= d_x
                u2 = f(x_vals)
                x_vals[j] -= d_x
                u4 = f(x_vals)
                x_vals[i] += d_x
                u3 = f(x_vals)
                x_vals[j] += d_x

                hessian[i][j] = (u1 - u2 - u3 + u4)/d_x**2
    return hessian


def check_conv(grad_f, eps):
    return np.sum(grad_f**2) < eps**2