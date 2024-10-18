export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTHONPATH=/NAS6/Members/linchenxi/dinov2
torchrun --nproc_per_node=6 dinov2/train/train.py --config-file="/NAS6/Members/linchenxi/dinov2/dinov2/configs/train/vitl16_short_s2.yaml" --output-dir="/NAS6/Members/linchenxi/projects/DINOV2/model3" --model_name="model-4" --no-resume