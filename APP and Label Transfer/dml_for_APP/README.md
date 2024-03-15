The script that utilizes distance metric learning with UMAP to project the unlabeled data (test) into the embeddings space built using labeled data (training). 
Produces the 3 CSV files with ground truth, APP and Phenograph labels and the UMAP coordinates.

Run example:

````
fn_in='../test_data/PHA0026.csv'

python dml_from_cmd.py $fn_in
````