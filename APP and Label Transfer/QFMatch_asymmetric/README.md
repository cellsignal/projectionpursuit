There are few variants of the same script.

If there is _custom modifier at the script name, it will be using the left file path and the right file path as input params, and it will be using the original file path and the comparison variant is there is no such modifier.

If there is no _no_black modifier at the script name, data points that are misclassified
will be visually highlighted by being painted black.

Run examples:

````
fn_in='../test_data/PHA0026.csv'
variant=0 # ground truth vs APP
# variant=1 # ground truth vs Phenograph
# variant=2 # Phenograph vs APP
ndim=10
bin_size=150
filter_size=300
workers_count=4

python QFMatch_parallel_from_cmd_asymmetric.py $fn_in $variant $bin_size $ndim $filter_size $workers_count

left_file='../dml/ground_truth_PHA0026.csv.csv'
right_file='../dml/projection_pursuit_PHA0026.csv.csv'

python QFMatch_parallel_from_cmd_custom_asymmetric_no_black.py $left_file $right_file $bin_size $ndim $filter_size  $workers_count
````