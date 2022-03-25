import numpy as np

def f(x1, x2):
    return 10*x1*x1 + 3*x1*x2 + x2*x2 + 10*x2


def calc_grad(f, n, x_vals):
    pass


def calc_hessian(f, n, x_vals, grad_f):
    pass


def calc_h(n, grad_f, hes):
    pass


def calc_next_x(h, hes, grad_f):
    pass


def check_conv(grad_f, eps):
    pass

n = 2
x = [np.array([1,1])]
eps = .001
grad_k = []
h_k = []

while True:
    grad_f = calc_grad(f, n, x[-1])
    if check_conv(grad_f, eps): break
    hes = calc_hessian(f, n, x[-1], grad_f[-1])
    h = calc_h(n, grad_f, hes)
    x.append(calc_next_x(h, hes, grad_f))

    grad_k.append(grad_f)
    h_k.append(h)


print(x[-1])
# TODO: functions, print results, visualize