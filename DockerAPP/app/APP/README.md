There are few variants of the same script.

If the script name contains the "_parallel" modifier, the parallel computations will be performed, otherwise consequent computations will be performed.

If the script name contains the "_fixed_param" modifier, it will use fixed intrinsic parameters, on the other hand, if the script contains "_dynamic_param" modifier, it will compute parameters at the runtime.

Run examples:

````
min_cluster_size=300
ndim=10

workers_count=4

fn_in='../test_data/PHA0026.csv'

python variation_parallel_from_cmd_fixed_param.py $fn_in $min_cluster_size $ndim $workers_count

python variation_from_cmd_dynamic_param.py $fn_in $min_cluster_size $ndim
````