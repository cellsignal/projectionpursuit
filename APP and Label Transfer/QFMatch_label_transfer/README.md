The **QFMatch_parallel_from_cmd_asymmetric_label_transfer.py** script assigns labels to the raw test data based on the clustered train data.

The **QFMatch_parallel_from_cmd_custom_asymmetric_with_noise_filtering.py** filters noise (if any), then executes many clusters-to-one cluster (asymmetric) matching with QFMatch between the test set and training set results of the distance metric learning script, and further  visualizes the matching outcomes and computes the misclassification
rate (if this task is required).

Run examples:

````
fn_in_train = 'dml/PHA1005.csv_orig_with_umap.csv'
fn_in_test_hdbscan = 'dml/PHA1005.csv_PHA0026.csv_hdbscan_with_umap.csv'
fn_in_test_orig = 'dml/PHA1005.csv_PHA0026.csv_orig_with_umap.csv'
fn_in_label_transfer_hdbscan = 'QFMatch/label_transfer_PHA1005.csv_orig_with_umap_PHA1005.csv_PHA0026.csv_hdbscan_with_umap.csv'

ndim=10
bin_size=150
filter_size=300

# step 1:
python QFMatch_parallel_from_cmd_asymmetric_label_transfer.py $fn_in_train $fn_in_test_hdbscan $bin_size $ndim

# step 2:
python QFMatch_parallel_from_cmd_custom_asymmetric_with_noise_filtering.py $fn_in_test_orig $fn_in_label_transfer_hdbscan $bin_size $ndim $filter_size
````