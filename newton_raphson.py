import numpy as np
from visualize import visualize_func_and_path
from common import f, calc_grad, is_pos_def, calc_hessian, check_conv
np.set_printoptions(precision=3)


def get_hessian(f, n, x_vals):
    hessian = calc_hessian(f, n, x_vals)
    if is_pos_def(hessian, n):
        return hessian
    else:
        return np.linalg.inv(np.identity(n)) # В формуле матрица обращается, поэтому нужно вернуть обратную


def calc_h(grad_f, hes, p):
    return grad_f.dot(p) / (hes.dot(p)).dot(p)


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
    hes = get_hessian(f, n, x[-1])
    p = np.linalg.inv(hes).dot(grad_f)
    p /= np.linalg.norm(p)
    h = calc_h(grad_f, hes, p)

    # Сохранение данных расчета
    grad_k.append(grad_f)
    h_k.append(h)
    f_k.append(f(x[-1]))

    # Сохранение следующей точки
    x.append(x[-1] - h*p)
    k += 1


# Вывод результатов
print('i\t{: >15s}\t{: >15s}{: >20s}{: >15s}'.format('(x1, x2)(k)', 'f(x_k)', 'grad_f(x_k)', 'h_k'))
for i in range(k):
    print('{}:\t{: >15s}\t{: >15.4f}{: >20s}{: >15.4f}'.format(i + 1, str(x[i]), f_k[i], str(grad_k[i]), h_k[i]))
print('\nResult: x={}\nf={}'.format(x[k], f(x[k])), end='\n\n')

visualize_func_and_path(f, (-3, 3), (-8, 2), x)
