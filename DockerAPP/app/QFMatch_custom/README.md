There is a version of symmetric QFMatch, for which numbers of cells for the test data and the train data may differ.

Run example:

````
ndim=10
bin_size=150
filter_size=300

left_file='../dml/ground_truth_PHA0026.csv.csv'
right_file='../dml/ground_truth_PHA1005.csv.csv'

python QFMatch_parallel_from_cmd_custom_different_left_and_right.py $left_file $right_file $bin_size $ndim

````