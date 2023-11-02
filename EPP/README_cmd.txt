Usage:

0. For macOS:
sh deps_python.sh

1. python variation_parallel_from_cmd.py fn_in min_cluster_size ndim
example:
python variation_parallel_from_cmd.py 1.csv 50 3

if variation_parallel_from_cmd.py is not working correctly, use variation_from_cmd.py instead

2. python dml_from_cmd.py fn_in
example:
python dml_from_cmd.py 1.csv

3. 
symmetric match:
python QFMatch_parallel_from_cmd.py fn_in variant bin_size ndim

asymmetric match:
QFMatch_parallel_from_cmd_asymmetric.py fn_in variant bin_size ndim

variant:
0 - ground_truth_ vs projection_pursuit_
1 - ground_truth_ vs phenograph_
2 - phenograph_ vs projection_pursuit_

example:
python QFMatch_parallel_from_cmd.py 1.csv 0 25 3

if QFMatch_parallel_from_cmd.py is not working correctly, use QFMatch_from_cmd.py instead