#!/bin/bash

torchrun --nproc_per_node=4 --master_port=9292 train.py \
    --train_data ./replicate_alpaca_data.json \
    --num_train_epochs 4 \
    --learning_rate 5e-4 \
    --train_batch_size 6 \
    --gradient_accumulation_steps 8 \
    --logging_steps 2 \
    --warmup_ratio 0.03 
