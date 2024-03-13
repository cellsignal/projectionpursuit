"""
The script utilizes distance metric learning with UMAP to project
the unlabeled data (test) into the embeddings space
built using labeled data (training).
"""

import sys
import os
import time

import csv
from pathlib import Path

import numpy as np

import umap
import phenograph

from sklearn import metrics

import numpy.core.defchararray as np_f

n_neighbors = 30
min_dist = 0.3
kNN = 30

fn_in_train = sys.argv[1]
fn_in = Path(fn_in_train).stem + '.csv'
fn_in_test = 'results/results_' + fn_in

fn_out_1 = 'dml/ground_truth_' + fn_in + '.csv'
fn_out_2 = 'dml/phenograph_' + fn_in + '.csv'
fn_out_3 = 'dml/projection_pursuit_' + fn_in + '.csv'
fn_score = 'dml/score_' + fn_in + '.txt'

if not os.path.exists('dml'):
    os.makedirs('dml')

with open(fn_in_train, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    train_data = np.array(list(reader))
    train_data = np_f.replace(train_data, '\'', '')
    train_data = train_data.astype(float)
train_data_for_calc = train_data[:, :-1]
train_labels = train_data[:, -1]
score1 = metrics.calinski_harabasz_score(train_data_for_calc, train_labels)
print(score1)
communities, graph, Q = phenograph.cluster(train_data_for_calc, n_jobs=1, k=kNN)
train_labels_2 = communities.astype(float)
score2 = metrics.calinski_harabasz_score(train_data_for_calc, train_labels_2)
print(score2)
t1 = time.time()
mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
train_embedding = mapper.fit_transform(train_data_for_calc, train_labels)
t2 = time.time()
print('training umap done ', t2 - t1)

headers.append('umap_x')
headers.append('umap_y')
result1 = np.column_stack((train_data_for_calc, train_labels, train_embedding))
(H, W) = train_data.shape
np.savetxt(fn_out_1, result1,
           fmt='%.5f',
           delimiter=',', header=','.join(headers), comments='')
result2 = np.column_stack((train_data_for_calc, train_labels_2, mapper.embedding_))
np.savetxt(fn_out_2, result2,
           fmt='%.5f'
           , delimiter=',', header=','.join(headers), comments='')
print('training done')

with open(fn_in_test, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers2 = next(reader)
    test_data = np.array(list(reader)).astype(float)
test_data_for_calc = test_data[:, :-1]
test_labels = test_data[:, -1]
score3 = metrics.calinski_harabasz_score(test_data_for_calc, test_labels)
print(score3)
t0 = time.time()
test_embedding = mapper.transform(test_data_for_calc)
t1 = time.time()

result3 = np.column_stack((test_data, test_embedding))
np.savetxt(fn_out_3, result3,
           fmt='%.5f'
           , delimiter=',', header=','.join(headers), comments='')
print('{0} done: {1}'.format(fn_out_3, t1 - t0))

with open(fn_score, 'w') as the_file:
    the_file.write('ground truth score: {0}\n'.format(score1))
    the_file.write('phenograph score: {0}\n'.format(score2))
    the_file.write('projection pursuit score: {0}\n'.format(score3))
the_file.close()
