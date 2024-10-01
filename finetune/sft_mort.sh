# export CUDA_VISIBLE_DEVICES=3,4,5,6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export ACCELERATE_LOG_LEVEL=info

accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes=6 \
    --main_process_port 1234 \
    finetune/run_sft_mortality.py \
    finetune/recipes/config_full_mortality.yaml \