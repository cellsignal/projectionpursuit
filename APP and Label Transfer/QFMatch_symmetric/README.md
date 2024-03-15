There are few variants of the same script.

If the script name contains the "_custom" modifier, it will utilize the left file path and the right file path as input parameters. On the other hand, if there is no such modifier, it will use the original file path, and the comparison variant will be determined accordingly.

If the script name contains the "_parallel" modifier, the parallel computations will be performed, otherwise consequent computations will be performed.

If the script name contains the "_with_clusters_visualization" modifier, it will additionally visualize the original train and test clusters.

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

python QFMatch_from_cmd.py $fn_in $variant $bin_size $ndim

python QFMatch_parallel_from_cmd_with_clusters_visualization.py $fn_in $variant $bin_size $ndim $workers_count

left_file='../dml/ground_truth_PHA0026.csv.csv'
right_file='../dml/projection_pursuit_PHA0026.csv.csv'

python QFMatch_from_cmd_custom_with_clusters_visualization.py $left_file $right_file $bin_size $ndim

python QFMatch_parallel_from_cmd_custom.py $left_file $right_file $bin_size $ndim $workers_count

````