export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTHONPATH=/NAS3/Members/linchenxi/dinov2
torchrun --nproc_per_node=6 dinov2/train/train_multimodal.py --config-file="/NAS6/Members/linchenxi/dinov2/dinov2/configs/train/vitl16_short_multimodal.yaml" --output-dir="/NAS3/Members/linchenxi/projects/DINOV2/models" --model_name="model-1" --no-resume