
#!/bin/bash

emb_reg_values=(0.0001)
time_reg_values=(0.01)
freq_reg_values=(1.0)
alpha_values=(10)

learning_rate_values=(0.05)
rank_values=(6000)
batch_size_values=(2000)

# 遍历所有可能的参数组合
for emb_reg in "${emb_reg_values[@]}"; do
    for time_reg in "${time_reg_values[@]}"; do
        for freq_reg in "${freq_reg_values[@]}"; do
            for learning_rate in "${learning_rate_values[@]}"; do
                for rank in "${rank_values[@]}"; do
                    for batch_size in "${batch_size_values[@]}"; do
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
                            python learner.py --valid_freq 10 --dataset GDELT --emb_reg $emb_reg --freq_reg $freq_reg \
                            --time_reg $time_reg --alpha $alpha  --learning_rate $learning_rate  \
                            --rank $rank  --batch_size $batch_size --max_epochs 51 --gpu 1
                        done
                    done
                done
            done
        done
    done
done