import time
import sys
import os

import csv
from pathlib import Path

import numpy as np

import umap

import numpy.core.defchararray as np_f

from sklearn.cluster import DBSCAN

n_neighbors = 30
min_dist = 0.3
kNN = 30

min_cluster_size = 10

fn_in_train = sys.argv[1]
fn_in_test_list_str = sys.argv[2]
fn_in_test_list = fn_in_test_list_str.split(',')

print('train: ', fn_in_train)
print('test list: ', fn_in_test_list)

if not os.path.exists('dml'):
    os.makedirs('dml')

fn_out_train = 'dml/' + Path(fn_in_train).stem + '.csv' + '_orig_with_umap.csv'

with open(fn_in_train, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    train_data = np.array(list(reader))
    train_data = np_f.replace(train_data, '\'', '')
data_for_calc_train = train_data[:, :-1].astype(float)
labels_train = train_data[:, -1].astype(float)
print('train labels: ', labels_train.shape)
t1 = time.time()
mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
embedding_train = mapper.fit_transform(data_for_calc_train, labels_train)
t2 = time.time()
print('training umap done ', t2 - t1)
headers.append('umap_x')
headers.append('umap_y')
result_train = np.column_stack((data_for_calc_train, labels_train, embedding_train))
np.savetxt(fn_out_train, result_train,
           fmt='%.5f',
           delimiter=',', header=','.join(headers), comments='')

for fn_in_test in fn_in_test_list:
    print('test: ', fn_in_test)

    fn_out_test = 'dml/' + Path(fn_in_train).stem + '.csv' + '_' + Path(fn_in_test).stem + '.csv' + '_DBSCAN_with_umap.csv'
    fn_out_test_orig = 'dml/' + Path(fn_in_train).stem + '.csv' + '_' + Path(fn_in_test).stem + '.csv' + '_orig_with_umap.csv'

    with open(fn_in_test, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        _headers = next(reader)
        data_test = np.array(list(reader))
        data_test = np_f.replace(data_test, '\'', '')
        data_test = data_test
    data_for_calc_test = data_test[:, :-1].astype(float)
    labels_test_orig = data_test[:, -1].astype(float)
    print('test labels: ', labels_test_orig.shape)

    (H,) = labels_test_orig.shape

    t3 = time.time()
    embedding_test = mapper.transform(data_for_calc_test)
    t4 = time.time()
    print('testing umap done ', fn_in_test, t4 - t3)
    labels_test = DBSCAN(eps=0.5, min_samples=1).fit_predict(embedding_test)
    t5 = time.time()
    print('testing DBSCAN done ', fn_in_test, t5 - t4)

    result_test = np.column_stack((data_for_calc_test, labels_test, embedding_test))
    np.savetxt(fn_out_test, result_test,
               fmt='%.5f',
               delimiter=',', header=','.join(headers), comments='')

    result_test_orig = np.column_stack((data_for_calc_test, labels_test_orig, embedding_test))
    np.savetxt(fn_out_test_orig, result_test_orig,
               fmt='%.5f',
               delimiter=',', header=','.join(headers), comments='')
