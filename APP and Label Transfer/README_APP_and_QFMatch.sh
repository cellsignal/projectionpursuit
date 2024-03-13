min_cluster_size=300
ndim=10
bin_size=150
filter_size=300

workers_count=4

# 271 KB:
fn_in=test_data/PHA0026.csv
# 1424 KB:
# fn_in = 'test_data/PHA1005.csv'
# 2421 KB:
# fn_in = 'test_data/PHA0036.csv'

# step 1:

# APP variant 1.A (parallel, with default workers count):
# python APP/variation_parallel_from_cmd_fixed_param.py $fn_in $min_cluster_size $ndim
# APP variant 1.B (parallel, with custom workers count):
python APP/variation_parallel_from_cmd_fixed_param.py $fn_in $min_cluster_size $ndim $workers_count
# APP variant 2 (consistent):
# python APP/variation_from_cmd_fixed_param.py $fn_in $min_cluster_size $ndim

# step 2:

# calculates phenograph and supervised UMAP by ground truth:
python dml_for_APP/dml_from_cmd.py $fn_in

fn_in="$(basename -- $fn_in)"

# step 3.A.A:
# asymmetric QFMatch with black misclassified cells:

# between ground truth and APP:
# with custom filter size and custom workers count:
python QFMatch_asymmetric/QFMatch_parallel_from_cmd_asymmetric.py $fn_in 0 $bin_size $ndim $filter_size $workers_count
# with custom filter size and default workers count:
# python QFMatch_asymmetric/QFMatch_parallel_from_cmd_asymmetric.py $fn_in 0 $bin_size $ndim $filter_size
# with default (zero) filter size and default workers count:
# python QFMatch_asymmetric/QFMatch_parallel_from_cmd_asymmetric.py $fn_in 0 $bin_size $ndim
# between ground truth and phenograph:
# with custom filter size:
python QFMatch_asymmetric/QFMatch_parallel_from_cmd_asymmetric.py $fn_in 1 $bin_size $ndim $filter_size

# step 3.A.B:
# asymmetric QFMatch without black misclassified cells:

# between ground truth and APP:
# with custom filter size and custom workers count:
python QFMatch_asymmetric/QFMatch_parallel_from_cmd_asymmetric_no_black.py $fn_in 0 $bin_size $ndim $filter_size $workers_count
# with custom filter size and default workers count:
# python QFMatch_asymmetric/QFMatch_parallel_from_cmd_asymmetric_no_black.py $fn_in 0 $bin_size $ndim $filter_size
# with default (zero) filter size and default workers count:
# python QFMatch_asymmetric/QFMatch_parallel_from_cmd_asymmetric_no_black.py $fn_in 0 $bin_size $ndim
# between ground truth and phenograph:
# with custom filter size:
python QFMatch_asymmetric/QFMatch_parallel_from_cmd_asymmetric_no_black.py $fn_in 1 $bin_size $ndim $filter_size

# step 3.B.A:
# symmetric QFMatch with black misclassified cells without clusters visualization:

# between ground truth and APP:
# python QFMatch_symmetric/QFMatch_parallel_from_cmd.py $fn_in 0 $bin_size $ndim
# between ground truth and phenograph:
# python QFMatch_symmetric/QFMatch_parallel_from_cmd.py $fn_in 1 $bin_size $ndim
# between phenograph and APP:
# python QFMatch_symmetric/QFMatch_parallel_from_cmd.py $fn_in 2 $bin_size $ndim

# step 3.B.B:
# symmetric QFMatch with black misclassified cells with clusters visualization:

# between ground truth and APP:
python QFMatch_symmetric/QFMatch_parallel_from_cmd_with_clusters_visualization.py $fn_in 0 $bin_size $ndim
# between ground truth and phenograph:
python QFMatch_symmetric/QFMatch_parallel_from_cmd_with_clusters_visualization.py $fn_in 1 $bin_size $ndim
# between phenograph and APP:
python QFMatch_symmetric/QFMatch_parallel_from_cmd_with_clusters_visualization.py $fn_in 2 $bin_size $ndim
