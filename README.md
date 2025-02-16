
### Installation
Create a conda environment with pytorch and scikit-learn :
```
conda create --name terdy_env python=3.8
source activate terdy_env
conda install --file requirements.txt -c pytorch
```


### Datasets

For complete datasets, you can download from: https://github.com/nk-ruiying/TCompoundE/tree/main/src_data

Use the following command to process the dataset:

```
python process_icews.py

python process_gdelt.py
```

This will create the files required to compute the filtered metrics.


### Reproducing results of TeRDy

In order to reproduce the results of TeRDy on the three datasets in the paper, run the following commands

```
bash run_TeRDy_GDELT.sh

bash run_TeRDy_ICEWS14.sh

bash run_TeRDy_ICEWS15.sh
```


Our training framework is primarily based on TeAST and TCompoundE from the papers "Teast: Temporal knowledge graph embedding via archimedean spiral timeline (ACL 2023)" and "Simple but Effective Compound Geometric Operations for Temporal Knowledge Graph Completion (ACL 2024)", respectively.