
### Installation
Create a conda environment with pytorch and scikit-learn :
```
conda create --name terdy_env python=3.8
source activate terdy_env
conda install --file requirements.txt -c pytorch
```


### Datasets

```
python process_icews.py

python process_gdelt.py
```

This will create the files required to compute the filtered metrics.

### Reproducing results of TeRDy

In order to reproduce the results of TeRDy on the three datasets in the paper,  run the following commands

```
bash GDELT.sh

bash run_ICEWS14.sh

bash run_ICEWS15.sh
```

