from math import sqrt

def f(x1, x2):
    return 10*x1*x1 + 3*x1*x2 + x2*x2 + 10*x2


def calc_x_i(x_0, i, j, n, m):
    if i == j:
        return x_0 + (sqrt(n + 1) - 1)/(n*sqrt(2)) * m
    else:
        return x_0 + (sqrt(n + 1) + n - 1)/(n*sqrt(2)) * m


def is_finished(f_x_c, f_vals, eps):
    # проверка окончания
    for i in range(n+1):
        if abs(f_vals[i] - f_x_c) > eps:
            return False
    return True


n = 2 # размерность задачи оптимизации
m = 1 # размер ребра симплекса
eps = 0.2 # точность поиска
x = [[] * (n + 1)] # вершины симплекса
f_vals = [0 * (n + 1)] # значения функции в вершинах

# инициализация вершин симплеса
x[0] = [0, 0] # x_0
f_vals[0] = f(*x[0])
for i in range(1, n + 1):
    x[i] = [calc_x_i(x[0][j], i, j, n, m) for j in range(n)]
    f_vals[i] = f(x[i])

while True:
    k = f_vals.index(max(f_vals)) # f_max

    x_c = [0 * (n + 1)] # центр масс
    for i in range(n + 1):
        if i == k: continue
        for j in range(n):
            x_c[j] += x[i][j]
    x_c = list(map(lambda x_i_c: x_i_c/n, x_c))

    # отражение x_k относительно x_c
    x_new = [2*x_c[i] - x[k][i] for i in range(n)]
    f_new = f(*x_new)

    if f_new >= f[k]:
        # операция редукции
        r = f_vals.index(min(f_vals))
        for i in range(n + 1):
            x_i = [x[r][j] + (x[i][j] - x[r][j])/2 for j in range(n)]
    else:
        x[k] = x_new
        f_vals[k] = f_new

    for i in range(n + 1):
        for j in range(n):
            x_c[j] += x[i][j]
    x_c = list(map(lambda x_i_c: x_i_c/n, x_c))
    f_x_c = f(*x_c)

    # проверка на завершение
    if is_finished(f_x_c, f_vals, eps): break

ans_ind = f_vals.index(min(f_vals))
print('Результат: x_min =', x[k], '\nЗначение функции: f(x_min) =', f_vals[k])
