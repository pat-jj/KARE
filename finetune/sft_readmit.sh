# export CUDA_VISIBLE_DEVICES=3,4,5,6
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7
export ACCELERATE_LOG_LEVEL=info

accelerate launch \
    --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
    --num_processes=7 \
    --main_process_port 1528 \
    finetune/run_sft_readmission.py \
    finetune/recipes/kelpie/config_full_readmission.yaml \