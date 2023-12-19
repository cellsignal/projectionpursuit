import sys
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
import csv
from operator import attrgetter
from sklearn import metrics
import numpy.core.defchararray as np_f

fn_in = sys.argv[1]
fn_out = 'results/results_' + fn_in

min_cluster_size = int(sys.argv[2])
# h = .01
# sigma = 3
dq = 0.02
# q2 = 0.1
ndim = int(sys.argv[3])
max_dj = 5
draw_all = False

cluster_results = []


# looks for the index of the closest value in the array:
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# не используется
def sign_change_count(list):
    count = 0
    for i in range(len(list) - 1):
        if list[i] * list[i + 1] < 0:
            count += 1
    return count


# не используется
def is_saddle_point(mat, i, j):
    list = []
    list.append(mat[i - 1, j - 1] - mat[i, j])
    list.append(mat[i, j - 1] - mat[i, j])
    list.append(mat[i + 1, j - 1] - mat[i, j])
    list.append(mat[i + 1, j] - mat[i, j])
    list.append(mat[i + 1, j + 1] - mat[i, j])
    list.append(mat[i, j + 1] - mat[i, j])
    list.append(mat[i - 1, j + 1] - mat[i, j])
    list.append(mat[i - 1, j] - mat[i, j])
    list.append(mat[i - 1, j - 1] - mat[i, j])
    return sign_change_count(list) == 4


# не используется
def find_saddle_points(mat):
    (H, W) = mat.shape
    saddle_points = []
    for i in np.arange(1, W - 1):
        for j in np.arange(1, H - 1):
            if is_saddle_point(mat, j, i):
                saddle_points.append((j, i))
    return saddle_points


class Result:
    xxx = None
    yyy = None
    nnn = 0
    zz1 = None
    err = 0.0
    sqsum = 0
    n = 0
    m = 0
    score = 0.0


class Projection:
    index1 = 0
    index2 = 0
    err = 0.0
    sqsum = 0
    score = 0.0
    x_min = 0.0
    x_max = 0.0
    y_min = 0.0
    y_max = 0.0
    xx = None
    yy = None
    zz = None
    data = None
    labels = None
    result = None


def sum_along_y_min(zz):
    return np.sum(zz[0, :])


def sum_along_y_max(zz):
    return np.sum(zz[-1, :])


# ищет экстремаль с параметром q:
# def minimize(xx, yy, zz, q, max_dj):
def minimize(xx, yy, zz, q, max_dj,zmx):
    err = 0.0
    xx_ = xx[0, :]
    (n,) = xx_.shape
    yy_ = yy[:, 0]
    (m,) = yy_.shape
    # "притяжение" к параболе
    zz0 = np.power(yy - (q * yy_[-1] + (1 - q) * yy_[0]), 2)
    
    # среднее значение параболы:                   
    zz0m=zz0.mean()

    # оптимальное q2:
    q2 = np.power(zmx, 0.5) / zz0m

    zz1 = zz + q2 * zz0
    ii = np.arange(n)
    jj = []
    jj.append(int(q * m))
    # для всех координат по x:
    for i in np.arange(n - 1):
        j = jj[i]
        # значения функции z:
        zzz = []
        # координата по y:
        jjj = []
        dj = [0]
        for ddj in range(max_dj):
            dj.append(ddj)
            dj.append(-ddj)
        dj.sort()
        # ищем координату по y, в которой значение z минимально:
        for ddj in dj:
            jjj.append(j + ddj)
            if 0 <= j + ddj <= m - 1:
                zzz.append((zz1[j, i] + zz1[j + ddj, i + 1]) / 2)
            else:
                zzz.append(999999999)
        z = min(zzz)
        k = zzz.index(z)
        jj.append(jjj[k])
        ddj = dj[k]
        # прибавляем значение z к интегралу:
        err += (zz[j, i] + zz[j + ddj, i + 1]) / 2
    # координаты экстремали:
    xxx = xx_[ii]
    yyy = yy_[jj]

    # создаем объект для результата:
    res = Result()
    res.xxx = xxx
    res.yyy = yyy
    res.zz1 = zz1
    res.err = err
    res.n = n
    res.m = m
    return res


# разделяет данные data на две части по сепаратриссе xxx, yyy:
# xxx, yyy - координаты сепаратриссы
def split_data_by_separatrix(xxx, yyy, data, index1, index2):
    # нужная проекция:
    X = data[:, [index1, index2]]
    # результаты разделения:
    split_results = []
    actual_data_1 = []
    actual_data_2 = []
    (H, W) = data.shape
    for j in range(H):
        k = find_nearest_idx(xxx, X[j, 0])
        if X[j, 1] < yyy[k]:
            actual_data_1.append(data[j, :].tolist())
        else:
            actual_data_2.append(data[j, :].tolist())
    if len(actual_data_1) > min_cluster_size and len(actual_data_2) > min_cluster_size:
        split_results.append(actual_data_1)
        split_results.append(actual_data_2)
        split_results_2 = np.row_stack((actual_data_1, actual_data_2))
        split_labels = np.array([0] * len(actual_data_1) + [1] * len(actual_data_2))
    else:
        split_results = [data]
        split_results_2 = data
        split_labels = np.array([0] * (len(actual_data_1) + len(actual_data_2)))
    return split_results, split_results_2, split_labels


# рисует результаты разделения:
def draw_split_results(step, proj, split_results):
    for i in range(len(split_results)):
        actual_data = split_results[i]
        # fig = plt.figure()
        # ax = fig.gca()
        plt.clf()
        ax = plt.gca()
        ax.set_xlim([proj.x_min, proj.x_max])
        ax.set_ylim([proj.y_min, proj.y_max])
        if len(actual_data) > 0:
            actual_data = np.array(actual_data)
            X = actual_data[:, [proj.index1, proj.index2]]
            (H, W) = X.shape
            labels = [0] * H
            labels = np.array(labels)
            xx = proj.x_min + proj.xx * (proj.x_max - proj.x_min)
            yy = proj.y_min + proj.yy * (proj.y_max - proj.y_min)
            ax.contourf(xx, yy, proj.zz, 30, alpha=0.4)
            x_ = proj.x_min + X[:, 0] * (proj.x_max - proj.x_min)
            y_ = proj.y_min + X[:, 1] * (proj.y_max - proj.y_min)
            ax.scatter(x_, y_, c=labels, alpha=0.8)
            xxx = proj.x_min + proj.result.xxx * (proj.x_max - proj.x_min)
            yyy = proj.y_min + proj.result.yyy * (proj.y_max - proj.y_min)
            ax.plot(xxx, yyy)
            ax.set_title('step: {0}; proj: {1}_{2}; score: {3}'.format(step, proj.index1, proj.index2, proj.score))
            plt.savefig('split/{0}({1})_{2}_{3}.png'.format(step, i + 1, proj.index1, proj.index2))
        # plt.close(fig)


# рисует результаты разделения:
def draw_split_results_2(step, proj, split_results):
    # fig = plt.figure()
    # ax = fig.gca()
    plt.clf()
    ax = plt.gca()
    ax.set_xlim([proj.x_min, proj.x_max])
    ax.set_ylim([proj.y_min, proj.y_max])
    xx = proj.x_min + proj.xx * (proj.x_max - proj.x_min)
    yy = proj.y_min + proj.yy * (proj.y_max - proj.y_min)
    ax.contourf(xx, yy, proj.zz, 30, alpha=0.4)
    for i in range(len(split_results)):
        actual_data = split_results[i]
        if len(actual_data) > 0:
            actual_data = np.array(actual_data)
            X = actual_data[:, [proj.index1, proj.index2]]
            (H, W) = X.shape
            labels = [i] * H
            labels = np.array(labels)
            x_ = proj.x_min + X[:, 0] * (proj.x_max - proj.x_min)
            y_ = proj.y_min + X[:, 1] * (proj.y_max - proj.y_min)
            ax.scatter(x_, y_, c=labels, alpha=0.8)
    xxx = proj.x_min + proj.result.xxx * (proj.x_max - proj.x_min)
    yyy = proj.y_min + proj.result.yyy * (proj.y_max - proj.y_min)
    ax.plot(xxx, yyy)
    ax.set_title('step: {0}; proj: {1}_{2}; score: {3}'.format(step, proj.index1, proj.index2, proj.score))
    plt.savefig('all_split/{0}_{1}_{2}.png'.format(step, proj.index1, proj.index2))
    # plt.close(fig)


# рисует проекцию:
def draw_projection(step, proj, data):
    # fig = plt.figure()
    # ax = fig.gca()
    plt.clf()
    ax = plt.gca()
    ax.set_xlim([proj.x_min, proj.x_max])
    ax.set_ylim([proj.y_min, proj.y_max])
    X = data[:, [proj.index1, proj.index2]]
    (H, W) = X.shape
    labels = [0] * H
    labels = np.array(labels)
    #ax.pcolor(proj.xx, proj.yy, proj.zz)
    #ax.contourf(proj.xx, proj.yy, proj.zz, 300, alpha=0.4)
    #ax.contourf(proj.xx, proj.yy, proj.zz, 300, alpha=0.4)
    #ax.plot(proj.result.xxx, proj.result.yyy)
    x_ = proj.x_min + X[:, 0] * (proj.x_max - proj.x_min)
    y_ = proj.y_min + X[:, 1] * (proj.y_max - proj.y_min)
    ax.scatter(x_, y_, c=labels, alpha=0.8)
    ax.set_title('step: {0}; proj: {1}_{2}'.format(step, proj.index1, proj.index2))
    plt.savefig('all_proj/{0}_{1}_{2}.png'.format(step, proj.index1, proj.index2))
    # plt.close(fig)


# рекурсивная итерация разделения:
def split_iteration(step, data_list, data_for_calc_0):
    # s: 0 или 1
    for s in range(len(data_list)):
        data_for_calc = np.array(data_list[s])

        # пропускаем шаг, если данных слишком мало:
        (W, H) = data_for_calc.shape
        if (W < 2 * min_cluster_size):
            cluster_results.append(data_for_calc)
            print('skipping step')
            continue

        # массив проекций:
        projections = []

        t0 = time.time()
        # пробегаем по всем индексам:
        for index1 in range(ndim):
            for index2 in range(ndim):
                if index2 > index1 or index2 < index1:
                    # данные с нормированными осями x, y:
                    X = data_for_calc[:, [index1, index2]]
                    # оригинальные данные:
                    X0 = data_for_calc_0[:, [index1, index2]]
                    x_min_0, x_max_0 = X0[:, 0].min(), X0[:, 0].max()
                    y_min_0, y_max_0 = X0[:, 1].min(), X0[:, 1].max()
                    x_min, x_max = 0, 1
                    y_min, y_max = 0, 1
                                                       
                    # N = int(1/h)
                    #  сумма элементов гистограммы:
                    szz=len(X0)
                    #  оптимальное число разбиений гистограммы:
                    N = int(np.power((szz-1)*(szz-1)*3/4, 0.2)*4)
                    h = 1/N
                    #  оптимальная ширина гаусса сглаживания:
                    sigma = 0.05 * N
                    
                    # строим сетку:
                    xedges, yedges = np.linspace(x_min, x_max, N + 1), np.linspace(y_min, y_max, N + 1)
                    # строим гистограмму:
                    hist, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], (xedges, yedges))
                    zz = hist
                    # строим плотность точек по гистограмме:
                    zz = gaussian(zz, sigma=(sigma, sigma), truncate=3.5)
                    zz = zz.T
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                    xx_ = xx[0, :]
                    (n,) = xx_.shape
                    yy_ = yy[:, 0]
                    (m,) = yy_.shape

                    # cреднее значение гистограммы: 
                    zmx=zz.mean()
                    
                    results0 = []
                    # для каждого значения параметра ищем экстремаль:
                    for q in np.arange(0, 1, dq):
                        # res = minimize(xx, yy, zz, q, max_dj)
                        res = minimize(xx, yy, zz, q, max_dj,zmx)
                        
                        results0.append(res)
                    results = []
                    for i in np.arange(1, len(results0) - 1):
                        res_l = results0[i - 1]
                        res_r = results0[i + 1]
                        res = results0[i]
                        # если ошибка по экстремали меньше, чем ошибка нижнего и верхнего соседа
                        # то это "настоящая" экстремаль
                        if res.err < res_l.err and res.err < res_r.err:
                            # добавляем к результатам по проекции:
                            results.append(res)
                            # пытаемся разделить данные по сепаратриссе:
                            split_results, split_results_2, split_labels = \
                                split_data_by_separatrix(res.xxx, res.yyy, data_for_calc, index1, index2)
                            # если успешно разделилось, ищем score:
                            if len(split_results) > 1:
                                res.score = metrics.calinski_harabasz_score(split_results_2, split_labels)
                            else:
                                res.score = 0.0
                    print('results found: {0}'.format(len(results)))

                    # создаем объект проекции:
                    proj = Projection()
                    proj.index1 = index1
                    proj.index2 = index2
                    proj.x_min = x_min_0
                    proj.x_max = x_max_0
                    proj.y_min = y_min_0
                    proj.y_max = y_max_0
                    proj.xx = xx
                    proj.yy = yy
                    proj.zz = zz
                    proj.data = data_for_calc

                    # рисуем проекцию, если опция включена:
                    if draw_all:
                        draw_projection(step * 10 + s + 1, proj, data_for_calc)

                    # если найдены результаты:
                    if len(results) > 0:
                        # ищем результат с максимальным score
                        min_res = max(results, key=attrgetter('score'))
                        proj.err = min_res.err
                        proj.score = min_res.score
                        proj.result = min_res
                        # добавляем проекцию в массив:
                        projections.append(proj)
                        print('projection: {0}, {1}, {2}, {3}'.format(step * 10 + s + 1, index1, index2, proj.score))
                        # рисуем, если включена опция:
                        if draw_all:
                            draw_projection(step * 10 + s + 1, proj, data_for_calc)
                            split_results, split_results_2, split_labels = split_data_by_separatrix(
                                proj.result.xxx, proj.result.yyy, proj.data, proj.index1, proj.index2)
                            if len(split_results) > 1:
                                draw_split_results_2(step * 10 + s + 1, proj, split_results)

        if len(projections) > 0:
            # ищем среди всех проекций с максимальным score:
            min_proj = max(projections, key=attrgetter('score'))
            index1 = min_proj.index1
            index2 = min_proj.index2
            xxx = min_proj.result.xxx
            yyy = min_proj.result.yyy
            data = min_proj.data
            # пробуем разделить:
            split_results, split_results_2, split_labels = split_data_by_separatrix(xxx, yyy, data, index1, index2)
            if len(split_results) > 1:
                print('{0}, {1}'.format(len(split_results[0]), len(split_results[1])))
                draw_split_results(step * 10 + s + 1, min_proj, split_results)
                t1 = time.time()
                print('iteration done ', t1 - t0)
                # успешно разделилось, идем на следующий шаг рекурсии:
                split_iteration(step * 10 + s + 1, split_results, data_for_calc_0)
            else:
                # не разделилось, добавляем результат к результатам кластеризации:
                cluster_results.append(split_results[0])
                t1 = time.time()
                print('iteration done ', t1 - t0)
        else:
            cluster_results.append(data_for_calc)
            t1 = time.time()
            print('iteration done ', t1 - t0)


# создаем и чистим папки:
if not os.path.exists('all_proj'):
    os.makedirs('all_proj')
dir = os.listdir('all_proj')
for file in dir:
    if file.endswith('.png'):
        os.remove('all_proj/' + file)
if not os.path.exists('all_split'):
    os.makedirs('all_split')
dir = os.listdir('all_split')
for file in dir:
    if file.endswith('.png'):
        os.remove('all_split/' + file)
if not os.path.exists('split'):
    os.makedirs('split')
dir = os.listdir('split')
for file in dir:
    if file.endswith('.png'):
        os.remove('split/' + file)

if not os.path.exists('results'):
    os.makedirs('results')

out_dir = Path(fn_out).parent.absolute()
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# считываем файл:
with open(fn_in, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    data = np.array(list(reader))
    data = np_f.replace(data, '\'', '')
    data = data.astype(float)
# "рабочие" данные:
data_for_calc = data[:, 0:ndim]
# оригинал данных:
data_for_calc_0 = np.copy(data_for_calc)
(H, W) = data_for_calc.shape
ii = np.arange(H)

for i in range(ndim):
    a = data_for_calc[:, i]
    a = np.interp(a, (a.min(), a.max()), (0, 1))
    # нормируем данные по x и y:
    data_for_calc[:, i] = a

data_for_calc = np.column_stack((data_for_calc, ii, data_for_calc_0))

headers = headers[0:ndim]
headers.append('cluster_id')

t0 = time.time()
# заходим на нулевой шаг рекурсии:
split_iteration(0, [data_for_calc], data_for_calc_0)
t1 = time.time()
print('splitting done ', t1 - t0)

# cluster_results - глобальная переменная
# собираем исходные данные и номера полученных кластеров в массив:
for i in range(len(cluster_results)):
    cluster_data = []
    for row in cluster_results[i]:
        row2 = np.append(row, i)
        cluster_data.append(row2)
    print(len(cluster_data))
    array = np.array(cluster_data)
    if i == 0:
        result_array = array
    else:
        result_array = np.row_stack([result_array, array])

# сохраняем результаты:
result_array = result_array[result_array[:, ndim].argsort()]
result_array = result_array[:, ndim+1:]

np.savetxt(fn_out, result_array,
           fmt='%s', header=','.join(headers),
           delimiter=',', comments='')
