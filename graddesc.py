import numpy as np
from visualize import draw_function_3dsurface_and_path
np.set_printoptions(precision=4)

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
    

def calc_h(f, grad_f, n, x_vals):
    hessian = calc_hessian(f, n, x_vals)
    return grad_f.dot(grad_f)/hessian.dot(grad_f).dot(grad_f)


def calc_next_x(x, h, grad):
    return x - h*grad


def check_eps(grad, eps):
    return np.sum(grad**2) < eps


n = 2
x = [np.ones(2)]
eps = .001

grad_f = []
h = []
k = 0
while True:
    grad_f.append(calc_grad(f, n, x[-1]))
    if check_eps(grad_f[-1], eps): break
    h.append(calc_h(f, grad_f[-1], n, x[-1]))
    x.append(calc_next_x(x[-1], h[-1], grad_f[-1]))
    k += 1

# log calculation process
print('i\t{: >20}\t{: >20}\t{})'.format('(x1, x2)(k)', 'grad_f(k)', 'h(k)'))
for i in range(k):
    print('{}:\t{: >20}\t{: >20}\t{:.4f}'.format(i + 1, str(x[i]), str(grad_f[i]), h[i]))
print('\nResult: x={}\nf={}'.format(x[k], f(x[k])))

# visualize function and path in space
draw_function_3dsurface_and_path(f, (-3, 3), (-8, 2), x)