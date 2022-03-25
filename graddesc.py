import numpy as np
from visualize import visualize_func_and_path
from common import f, calc_grad, calc_hessian, check_conv
np.set_printoptions(precision=4)
    

def calc_h(f, grad_f, n, x_vals):
    hessian = calc_hessian(f, n, x_vals)
    return grad_f.dot(grad_f)/hessian.dot(grad_f).dot(grad_f)


def calc_next_x(x, h, grad):
    return x - h*grad


n = 2
x = [np.ones(2)]
eps = .001

grad_f = []
h = []
k = 0
while True:
    grad_f.append(calc_grad(f, n, x[-1]))
    if check_conv(grad_f[-1], eps): break
    h.append(calc_h(f, grad_f[-1], n, x[-1]))
    x.append(calc_next_x(x[-1], h[-1], grad_f[-1]))
    k += 1

# log calculation process
print('i\t{: >20}\t{: >20}\t{})'.format('(x1, x2)(k)', 'grad_f(k)', 'h(k)'))
for i in range(k):
    print('{}:\t{: >20}\t{: >20}\t{:.4f}'.format(i + 1, str(x[i]), str(grad_f[i]), h[i]))
print('\nResult: x={}\nf={}'.format(x[k], f(x[k])))

# visualize function and path in space
visualize_func_and_path(f, (-3, 3), (-8, 2), x)