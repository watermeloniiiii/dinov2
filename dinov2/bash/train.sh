export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export PYTHONPATH=/NAS3/Members/linchenxi/dinov2
torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0\
    --nproc_per_node=7\
    dinov2/train/train_multimodal.py \
    --config-file="/NAS3/Members/linchenxi/dinov2/dinov2/configs/train/vitl16_short_multimodal.yaml" \
    --output-dir="/NAS3/Members/linchenxi/projects/DINOV2/model_multimodal_1" \
    --model_name="model_multimodal_1" \
    --no-resume
