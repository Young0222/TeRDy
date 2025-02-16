
#!/bin/bash

emb_reg_values=(0.002)
time_reg_values=(0.1)
freq_reg_values=(0.0005)
alpha_values=(10)

learning_rate_values=(0.008)
rank_values=(8000)
batch_size_values=(6000)


for emb_reg in "${emb_reg_values[@]}"; do
    for time_reg in "${time_reg_values[@]}"; do
        for learning_rate in "${learning_rate_values[@]}"; do
            for rank in "${rank_values[@]}"; do
                for batch_size in "${batch_size_values[@]}"; do
                    for freq_reg in "${freq_reg_values[@]}"; do
                        for alpha in "${alpha_values[@]}"; do
                            echo "Running experiment with parameters:"
                            echo "  emb_reg      = $emb_reg"
                            echo "  time_reg     = $time_reg"
                            echo "  freq_reg     = $freq_reg"
                            echo "  alpha        = $alpha"
                            echo "  learning_rate = $learning_rate"
                            echo "  rank         = $rank"
                            echo "  batch_size   = $batch_size"
                            echo "--------------------------------------"

                            # 运行 Python 脚本
                            python learner.py --valid_freq 20 --dataset ICEWS05-15 \
                            --emb_reg $emb_reg  --time_reg $time_reg --freq_reg $freq_reg --alpha $alpha \
                            --learning_rate $learning_rate --rank $rank  --batch_size $batch_size  --max_epochs 101 --gpu 5
                        done
                    done
                done
            done
        done
    done
done