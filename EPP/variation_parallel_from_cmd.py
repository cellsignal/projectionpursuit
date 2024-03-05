import sys
import os
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.filters import gaussian
import csv
from operator import attrgetter
from sklearn import metrics
import numpy.core.defchararray as np_f
from multiprocessing import Pool

MAX_WORKERS = 8

start_time = time.time()

fn_in = sys.argv[1]
fn_out = 'results/results_' + fn_in
fn_out_txt = 'split_projections.txt'

min_cluster_size = int(sys.argv[2])
# h = .01
# sigma = 3
sigma = 5
dq = 0.02
# q2 = 0.1
betta = 0.1
ndim = int(sys.argv[3])
max_dj = 5
draw_all = True

draw_split_point_size = 3.0

cluster_results = []
iteration_paramList = []

result_projections = []

spectral = cm.get_cmap('Spectral', 2)
colors = spectral(np.linspace(0, 1, 2))
red = colors[0, :]
blue = colors[1, :]


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def sign_change_count(list):
    count = 0
    for i in range(len(list) - 1):
        if list[i] * list[i + 1] < 0:
            count += 1
    return count


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


def find_saddle_points(mat):
    (H, W) = mat.shape
    saddle_points = []
    for i in np.arange(1, W - 1):
        for j in np.arange(1, H - 1):
            if is_saddle_point(mat, j, i):
                saddle_points.append((j, i))
    return saddle_points

def SFun(X, y):
    n_clusters = len(np.unique(y))
    n_samples = X.shape[0]
    overall_mean = X.mean(axis=0)

    centroid = np.zeros((n_clusters, X.shape[1]))
    ssb, ssw = 0, 0
    for n_cluster in range(n_clusters):
        i = np.where(y==n_cluster)[0]
        ni = len(i)
        centroid[n_cluster, :] = X[i].mean(axis=0)
        ssb += ni * np.linalg.norm(centroid[n_cluster] - overall_mean)**2
        for xx in X[i, :]:
            ssw += np.linalg.norm(centroid[n_cluster] - xx)**2 / ni
    SF = ssb / ssw / n_samples / n_clusters
    return SF
    
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
    split_results = None


class Projection:
    step = 0
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


def minimize(xx, yy, zz, q, max_dj):
    err = 0.0
    xx_ = xx[0, :]
    (n,) = xx_.shape
    yy_ = yy[:, 0]
    (m,) = yy_.shape
    zz0 = np.power(yy - (q * yy_[-1] + (1 - q) * yy_[0]), 2)
    #print(zz.shape)
    #print(zz0.shape)
    q2 = (zz.max()-zz.min()) / zz0.max() * betta
    zz1 = zz + q2 * zz0
    ii = np.arange(n)
    jj = []
    jj.append(int(q * m))
    for i in np.arange(n - 1):
        j = jj[i]
        zzz = []
        jjj = []
        dj = [0]
        for ddj in range(max_dj):
            dj.append(ddj)
            dj.append(-ddj)
        dj.sort()
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
        err += (zz[j, i] + zz[j + ddj, i + 1]) / 2
    xxx = xx_[ii]
    yyy = yy_[jj]

    res = Result()
    res.xxx = xxx
    res.yyy = yyy
    res.zz1 = zz1
    res.err = err
    res.n = n
    res.m = m
    return res


def split_data_by_separatrix(xxx, yyy, data, index1, index2):
    X = data[:, [index1, index2]]
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
    #print('{0}, {1}'.format(len(actual_data_1), len(actual_data_2)))
    return split_results, split_results_2, split_labels


def draw_split_results(step, proj, split_results):
    if len(split_results) == 2:
        actual_data_1 = split_results[0]
        actual_data_1 = np.array(actual_data_1)
        X1 = actual_data_1[:, [proj.index1, proj.index2]]
        (H1, _) = X1.shape
        labels1 = [0] * H1
        actual_data_2 = split_results[1]
        actual_data_2 = np.array(actual_data_2)
        X2 = actual_data_2[:, [proj.index1, proj.index2]]
        (H2, _) = X2.shape
        labels2 = [1] * H2
        X = np.row_stack((X1, X2))
        labels = np.array(labels1 + labels2)

        plt.clf()
        ax = plt.gca()
        ax.set_xlim([proj.x_min, proj.x_max])
        ax.set_ylim([proj.y_min, proj.y_max])
        xx = proj.x_min + proj.xx * (proj.x_max - proj.x_min)
        yy = proj.y_min + proj.yy * (proj.y_max - proj.y_min)
        ax.contourf(xx, yy, proj.zz, 30, alpha=0.15)
        x_ = proj.x_min + X[:, 0] * (proj.x_max - proj.x_min)
        y_ = proj.y_min + X[:, 1] * (proj.y_max - proj.y_min)
        ax.scatter(x_, y_, c=labels, s=draw_split_point_size, cmap='Spectral', alpha=0.8)
        xxx = proj.x_min + proj.result.xxx * (proj.x_max - proj.x_min)
        yyy = proj.y_min + proj.result.yyy * (proj.y_max - proj.y_min)
        ax.plot(xxx, yyy)
        ax.set_title('step: {0}; proj: {1}_{2}; score: {3}'.format(step, proj.index1, proj.index2, proj.score))
        plt.savefig('split/{0}_{1}_{2}.png'.format(step, proj.index1, proj.index2))

        plt.clf()
        ax = plt.gca()
        ax.set_xlim([proj.x_min, proj.x_max])
        ax.set_ylim([proj.y_min, proj.y_max])
        x1_ = proj.x_min + X1[:, 0] * (proj.x_max - proj.x_min)
        y1_ = proj.y_min + X1[:, 1] * (proj.y_max - proj.y_min)
        ax.scatter(x1_, y1_, color=red, s=draw_split_point_size, alpha=0.8)
        plt.savefig('split/{0}_{1}_{2}(1).png'.format(step, proj.index1, proj.index2))

        plt.clf()
        ax = plt.gca()
        ax.set_xlim([proj.x_min, proj.x_max])
        ax.set_ylim([proj.y_min, proj.y_max])
        x2_ = proj.x_min + X2[:, 0] * (proj.x_max - proj.x_min)
        y2_ = proj.y_min + X2[:, 1] * (proj.y_max - proj.y_min)
        ax.scatter(x2_, y2_, color=blue, s=draw_split_point_size, alpha=0.8)
        plt.savefig('split/{0}_{1}_{2}(2).png'.format(step, proj.index1, proj.index2))


def draw_split_results_2(step, proj, split_results):
    if len(split_results) == 2:
        actual_data_1 = split_results[0]
        actual_data_1 = np.array(actual_data_1)
        X1 = actual_data_1[:, [proj.index1, proj.index2]]
        (H1, _) = X1.shape
        labels1 = [0] * H1
        actual_data_2 = split_results[1]
        actual_data_2 = np.array(actual_data_2)
        X2 = actual_data_2[:, [proj.index1, proj.index2]]
        (H2, _) = X2.shape
        labels2 = [1] * H2
        X = np.row_stack((X1, X2))
        labels = np.array(labels1 + labels2)

        plt.clf()
        ax = plt.gca()
        ax.set_xlim([proj.x_min, proj.x_max])
        ax.set_ylim([proj.y_min, proj.y_max])
        xx = proj.x_min + proj.xx * (proj.x_max - proj.x_min)
        yy = proj.y_min + proj.yy * (proj.y_max - proj.y_min)
        ax.contourf(xx, yy, proj.zz, 30, alpha=0.15)
        x_ = proj.x_min + X[:, 0] * (proj.x_max - proj.x_min)
        y_ = proj.y_min + X[:, 1] * (proj.y_max - proj.y_min)
        ax.scatter(x_, y_, c=labels, s=draw_split_point_size, cmap='Spectral', alpha=0.8)
        xxx = proj.x_min + proj.result.xxx * (proj.x_max - proj.x_min)
        yyy = proj.y_min + proj.result.yyy * (proj.y_max - proj.y_min)
        ax.plot(xxx, yyy)
        ax.set_title('step: {0}; proj: {1}_{2}; score: {3}'.format(step, proj.index1, proj.index2, proj.score))
        plt.savefig('all_split/{0}_{1}_{2}.png'.format(step, proj.index1, proj.index2))

        plt.clf()
        ax = plt.gca()
        ax.set_xlim([proj.x_min, proj.x_max])
        ax.set_ylim([proj.y_min, proj.y_max])
        x1_ = proj.x_min + X1[:, 0] * (proj.x_max - proj.x_min)
        y1_ = proj.y_min + X1[:, 1] * (proj.y_max - proj.y_min)
        ax.scatter(x1_, y1_, color=red, s=draw_split_point_size, alpha=0.8)
        plt.savefig('all_split/{0}_{1}_{2}(1).png'.format(step, proj.index1, proj.index2))

        plt.clf()
        ax = plt.gca()
        ax.set_xlim([proj.x_min, proj.x_max])
        ax.set_ylim([proj.y_min, proj.y_max])
        x2_ = proj.x_min + X2[:, 0] * (proj.x_max - proj.x_min)
        y2_ = proj.y_min + X2[:, 1] * (proj.y_max - proj.y_min)
        ax.scatter(x2_, y2_, color=blue, s=draw_split_point_size, alpha=0.8)
        plt.savefig('all_split/{0}_{1}_{2}(2).png'.format(step, proj.index1, proj.index2))

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
    #ax.plot(proj.result.xxx, proj.result.yyy)
    x_ = proj.x_min + X[:, 0] * (proj.x_max - proj.x_min)
    y_ = proj.y_min + X[:, 1] * (proj.y_max - proj.y_min)
    ax.scatter(x_, y_, c=labels, s=draw_split_point_size, alpha=0.8)
    ax.set_title('step: {0}; proj: {1}_{2}'.format(step, proj.index1, proj.index2))
    plt.savefig('all_proj/{0}_{1}_{2}.png'.format(step, proj.index1, proj.index2))
    # plt.close(fig)


def single_projection(param):
    data_for_calc = param[0]
    index1 =  param[1]
    index2 = param[2]
    step = param[3]
    s = param[4]
    data_for_calc_0 = param[5]

    X = data_for_calc[:, [index1, index2]]
    X0 = data_for_calc_0[:, [index1, index2]]
    x_min_0, x_max_0 = X0[:, 0].min(), X0[:, 0].max()
    y_min_0, y_max_0 = X0[:, 1].min(), X0[:, 1].max()
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    # N = int(1 / h)
    szz=len(X)
    N = 2*sigma*int(np.power((szz-1)*(szz-1)*3/4, 0.1)*2)
    h = 1/N
    xedges, yedges = np.linspace(x_min, x_max, N + 1), np.linspace(y_min, y_max, N + 1)
    # print(xedges)
    hist, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], (xedges, yedges))
    # hist, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], N)
    zz = hist
    #zz = gaussian(zz, sigma=(sigma, sigma), truncate=3.5)
    zz = gaussian(zz, sigma=(sigma, sigma), truncate=sigma)
    zz = zz.T
    # print(zz.shape)
    #xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
    xx_ = xx[0, :]
    (n,) = xx_.shape
    yy_ = yy[:, 0]
    (m,) = yy_.shape

    results0 = []
    for q in np.arange(0, 1, dq):
        res = minimize(xx, yy, zz, q, max_dj)
        results0.append(res)
    results = []
    for i in np.arange(1, len(results0) - 1):
        res_l = results0[i - 1]
        res_r = results0[i + 1]
        res = results0[i]
        # print(res.err)
        if res.err < res_l.err and res.err < res_r.err:
            results.append(res)
            split_results, split_results_2, split_labels = \
                split_data_by_separatrix(res.xxx, res.yyy, data_for_calc, index1, index2)
            res.split_results = split_results
            if len(split_results) > 1:
                # res.score = metrics.calinski_harabasz_score(split_results_2, split_labels)
                res.score = SFun(split_results_2, split_labels)
            else:
                res.score = 0.0
    print('results found: {0}'.format(len(results)))

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

    if draw_all:
        draw_projection(step * 10 + s + 1, proj, data_for_calc)

    if len(results) > 0:
        # min_res = min(results, key=attrgetter('err'))
        min_res = max(results, key=attrgetter('score'))
        proj.err = min_res.err
        proj.score = min_res.score
        proj.result = min_res
        proj.step = step * 10 + s + 1
        if draw_all:
            draw_split_results_2(step * 10 + s + 1, proj, min_res.split_results)
        print('projection: {0}, {1}, {2}, {3}'.format(step * 10 + s + 1, index1, index2, proj.score))
        return proj
    else:
        return None


def split_iteration(step, data_list, data_for_calc_0):
    for s in range(len(data_list)):
        data_for_calc = np.array(data_list[s])

        (W, H) = data_for_calc.shape
        if (W < 2 * min_cluster_size):
            cluster_results.append(data_for_calc)
            print('skipping step')
            continue

        projections = []

        t0 = time.time()
        with Pool(MAX_WORKERS) as p:
            paramList = []
            for index1 in range(ndim):
                for index2 in range(ndim):
                    if index2 > index1 or index2 < index1:
                        paramList.append([data_for_calc, index1, index2, step, s, data_for_calc_0])
            resultList = p.map(single_projection, paramList)
            for result in resultList:
                if not result is None:
                    projections.append(result)

        if len(projections) > 0:
            #min_proj = min(projections, key=attrgetter('err'))
            min_proj = max(projections, key=attrgetter('score'))
            index1 = min_proj.index1
            index2 = min_proj.index2
            xxx = min_proj.result.xxx
            yyy = min_proj.result.yyy
            data = min_proj.data
            split_results, split_results_2, split_labels = split_data_by_separatrix(xxx, yyy, data, index1, index2)
            if len(split_results) > 1:
                print('{0}, {1}'.format(len(split_results[0]), len(split_results[1])))
                draw_split_results(step * 10 + s + 1, min_proj, split_results)
                global iteration_paramList
                iteration_paramList.append([step * 10 + s + 1, split_results, data_for_calc_0])
                global result_projections
                result_projections.append([min_proj.step, min_proj.index1, min_proj.index2])
            else:
                cluster_results.append(split_results[0])
        else:
            cluster_results.append(data_for_calc)
        t1 = time.time()
        print('iteration done ', t1 - t0)


if __name__ == '__main__':
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
    print(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(fn_in, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data = np.array(list(reader))
        data = np_f.replace(data, '\'', '')
        data = data.astype(float)
    data_for_calc = data[:, 0:ndim]
    data_for_calc_0 = np.copy(data_for_calc)

    (H, W) = data_for_calc.shape

    draw_split_point_size = 3.0 * 1000 / H

    ii = np.arange(H)

    for i in range(ndim):
        a = data_for_calc[:, i]
        a = np.interp(a, (a.min(), a.max()), (0, +1))
        data_for_calc[:, i] = a

    data_for_calc = np.column_stack((data_for_calc, ii, data_for_calc_0))

    headers = headers[0:ndim]
    headers.append('cluster_id')

    t0 = time.time()
    iteration_paramList.append([0, [data_for_calc]])
    iter_number = 0
    while iter_number < len(iteration_paramList):
        print('iter_number', iter_number)
        param = iteration_paramList[iter_number]
        step = param[0]
        data_list = param[1]
        split_iteration(step, data_list, data_for_calc_0)
        iter_number += 1
    t1 = time.time()
    print('splitting done ', t1 - t0)

    print('')    
    print('number of clusters = ', len(cluster_results))
    
    for i in range(len(cluster_results)):
        cluster_data = []
        for row in cluster_results[i]:
            row2 = np.append(row, i)
            cluster_data.append(row2)
        # print(len(cluster_data))
        print('size of cluster', i, ' = ', len(cluster_data))
        array = np.array(cluster_data)
        if i == 0:
            result_array = array
        else:
            result_array = np.row_stack([result_array, array])

    result_array = result_array[result_array[:, ndim].argsort()]
    result_array = result_array[:, ndim + 1:]

    np.savetxt(fn_out, result_array,
               fmt='%s', header=','.join(headers),
               delimiter=',', comments='')

    if len(cluster_results) > 1:
        #sc = metrics.calinski_harabasz_score(result_array, result_array[:,ndim])
        sc = SFun(result_array, result_array[:,ndim])
    else:
        sc = 0.0

    # print('sigma = ', sigma)
    # print('betta = ', betta)
    # print('Calinski-Harabasz score = ', sc)

    # end time
    end_time = time.time()

    # elapsed time
    elapsed_time = end_time - start_time
    print('total elapsed time: ', elapsed_time, 's')

    with open(fn_out_txt, 'w') as txt_file:
        txt_file.write('step\tindex1\tindex2\n')
        for proj in result_projections:
            txt_file.write('{0}\t{1}\t{2}\n'.format(proj[0], proj[1], proj[2]))
    txt_file.close()
