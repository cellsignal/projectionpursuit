import glob
import subprocess
from pathlib import Path

ndim = 10
bin_size = 150
filter_size = 150

fn_in_0 = 'test_data/PHA1005.csv'
fn_in_list = ['test_data/PHA0026.csv', 'test_data/PHA0036.csv']

fn_in_list_str = ','.join(fn_in_list)

# step 1:
subprocess.run(["python", "dml_for_label_transfer/dml_label_transfer_from_cmd_SVC.py", fn_in_0, fn_in_list_str])

for fn_in in fn_in_list:
    print('file: ', fn_in)
    fn_in_train = 'dml/' + Path(fn_in_0).stem + '.csv' + '_orig_with_umap.csv'
    fn_in_test_SVC = 'dml/' + Path(fn_in_0).stem + '.csv' + '_' + Path(fn_in).stem + '.csv' + '_SVC_with_umap.csv'
    fn_in_label_transfer_SVC = 'QFMatch/label_transfer_' + Path(fn_in_train).stem + '_' + Path(fn_in_test_SVC).stem + '.csv'
    fn_in_test_orig = 'dml/' + Path(fn_in_0).stem + '.csv' + '_' + Path(fn_in).stem + '.csv' + '_orig_with_umap.csv'
    # step 2:
    subprocess.run(
        ["python", "QFMatch_label_transfer/QFMatch_parallel_from_cmd_asymmetric_label_transfer.py",
         fn_in_train, fn_in_test_SVC,
         str(bin_size), str(ndim)])
    # step 3:
    subprocess.run(
        ["python", "QFMatch_label_transfer/QFMatch_parallel_from_cmd_custom_asymmetric_with_noise_filtering.py",
         fn_in_test_orig, fn_in_label_transfer_SVC,
         str(bin_size), str(ndim), str(filter_size)])

