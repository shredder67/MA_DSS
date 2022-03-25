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
    for f_i in f_vals:
        if abs(f_i - f_x_c) >= eps:
            return False
    return True


def print_x(i, x, f_val):
    print(str(i),':',end='\t')
    for x_j in x:
        print("{:.4f}".format(x_j),end='\t')
    print("{:.4f}".format(f_val))


def print_table(x, f_vals, n, header=""):
    print(header)
    print('----'*n)
    for i in range(n):
        print_x(i, x[i], f_vals[i])
    print('----'*n, end='\n\n')


n = 2 # размерность задачи оптимизации
m = 10.0 # размер ребра симплекса
eps = 0.001 # точность поиска
x = [[] for _ in range(n + 1)] # вершины симплекса
f_vals = [0] * (n + 1) # значения функции в вершинах

# инициализация вершин симплеса
x[0] = [1, 1] # x_0
f_vals[0] = f(*x[0])
for i in range(1, n + 1):
    x[i] = [calc_x_i(x[0][j], i, j, n, m) for j in range(n)]
    f_vals[i] = f(*x[i])

print_table(x, f_vals, n + 1, "Начальная конфигурация")

counter = n + 1


while True:
    k = f_vals.index(max(f_vals)) # f_max

    x_c = [0] * n # центр масс
    for i in range(n + 1):
        if i == k: continue
        for j in range(n):
            x_c[j] += x[i][j]
    x_c = list(map(lambda x_c_i: x_c_i/n, x_c))

    # отражение x_k относительно x_c
    x_new = [2*x_c[i] - x[k][i] for i in range(n)]
    f_new = f(*x_new)

    if f_new >= f_vals[k]:
        # операция редукции
        r = f_vals.index(min(f_vals))
        for i in range(n + 1):
            if i == r: continue 
            x_i = [x[r][j] + (x[i][j] - x[r][j])/2 for j in range(n)]
            x[i] = x_i
    else:
        # print_x(counter, x_new, f_new)
        counter += 1
        x[k] = x_new
        f_vals[k] = f_new
    
    print_table(x, f_vals, n + 1, "Итерация {}".format(counter))

    x_c = [0] * n
    for i in range(n + 1):
        for j in range(n):
            x_c[j] += x[i][j]
    x_c = list(map(lambda x_c_i: x_c_i/(n + 1), x_c))
    f_x_c = f(*x_c)

    # проверка на завершение
    if is_finished(f_x_c, f_vals, eps): break

ans_ind = f_vals.index(min(f_vals))
print('Результат: x_min =', x[k], '\nЗначение функции: f(x_min) =', f_vals[k])