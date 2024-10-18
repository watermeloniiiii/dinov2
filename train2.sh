export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=/NAS6/Members/linchenxi/dinov2
torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0\
    --nproc_per_node=4\
    dinov2/train/train.py --config-file="/NAS6/Members/linchenxi/dinov2/dinov2/configs/train/vitl16_short_s2.yaml" --output-dir="/NAS6/Members/linchenxi/projects/DINOV2/model8" --model_name="model-8" --no-resume
