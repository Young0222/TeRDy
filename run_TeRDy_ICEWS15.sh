
#!/bin/bash

python learner.py --valid_freq 20 --dataset ICEWS05-15 --emb_reg 0.002  --time_reg 0.1 --freq_reg 0.0005 --alpha 10 \
        --learning_rate 0.008 --rank 8000  --batch_size 6000  --max_epochs 101 --gpu 5