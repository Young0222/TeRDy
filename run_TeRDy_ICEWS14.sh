
#!/bin/bash

python learner.py --valid_freq 20 --dataset ICEWS14 --emb_reg 0.005 --time_reg 0.005 --freq_reg 0.0005 --alpha 10 \
        --learning_rate 0.02 --rank 6000 --batch_size 4000 --max_epochs 101 --gpu 4 \