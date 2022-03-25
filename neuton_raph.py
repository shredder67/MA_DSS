import numpy as np
from visualize import visualize_func_and_path
np.set_printoptions(precision=3)

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
        return np.delete(np.delete(arr,i,axis=0), j, axis=1)

    # Все миноры должны быть положительны
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

    if is_pos_def(hessian, n):
        return hessian
    else:
        return np.linalg.inv(np.identity(n)) # В формуле матрица обращается, поэтому нужно вернуть обратную


def calc_h(grad_f, hes, p):
    return grad_f.dot(p) / hes.dot(p).dot(p)


def check_conv(grad_f, eps):
    return np.sum(grad_f**2) < eps**2


n = 2
x = [np.ones(2)]
eps = .001

f_k = []
grad_k = []
h_k = []

k = 0
while True:
    # Расчет значений градиента, Гессиана и параметра h
    grad_f = calc_grad(f, n, x[-1])
    if check_conv(grad_f, eps): break
    hes = calc_hessian(f, n, x[-1])
    p = np.linalg.inv(hes).dot(grad_f)
    h = calc_h(grad_f, hes, p)

    # Сохранение данных расчета
    grad_k.append(grad_f)
    h_k.append(h)
    f_k.append(f(x[-1]))

    # Сохранение следующей точки
    x.append(x[-1] - h*p)
    k += 1


# Вывод результатов
print('i\t{: >15s}\t{: >15s}{: >20s})'.format('(x1, x2)(k)', 'f(x_k)', 'grad_f(x_k)'))
for i in range(k):
    print('{}:\t{: >15s}\t{: >15.4f}{: >20s}'.format(i + 1, str(x[i]), f_k[i], str(grad_k[i])))
print('\nResult: x={}\nf={}'.format(x[k], f(x[k])), end='\n\n')

visualize_func_and_path(f, (-3, 3), (-8, 2), x)
