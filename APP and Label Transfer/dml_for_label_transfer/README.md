There are few variants of the same script.

The script utilizes distance metric learning with UMAP to project the unlabeled data (test) into the embeddings space
built using labeled data (training) and assigns the labels to the test data with different clusterers: hdbscan, DBSCAN and SVC.

Run example:

````
fn_in_0='../test_data/PHA1005.csv'
fn_in_list='../test_data/PHA0026.csv,../test_data/PHA0036.csv'

python dml_label_transfer_from_cmd_SVC.py $fn_in_0 $fn_in_list
````