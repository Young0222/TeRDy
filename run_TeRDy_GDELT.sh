
#!/bin/bash

python learner.py --valid_freq 10 --dataset GDELT --emb_reg 0.0001 --freq_reg 1.0 --time_reg 0.01 --alpha 10  \
        --learning_rate 0.05  --rank 6000  --batch_size 2000 --max_epochs 51 --gpu 1