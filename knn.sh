export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=/NAS6/Members/linchenxi/dinov2
torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0\
    --nproc_per_node=4\
    "/NAS6/Members/linchenxi/dinov2/dinov2/eval/knn.py" \
    --config-file="/NAS6/Members/linchenxi/projects/DINOV2/model8/config.yaml"\
    --pretrained-weights="/NAS6/Members/linchenxi/projects/DINOV2/model8/eval/training_24999/teacher_checkpoint.pth"\
    --train-dataset="ClusterSentinel2:split=TEST:root=/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/clusters:extra=/NAS6/Members/linchenxi/ILSVRC"\
    --val-dataset="ClusterSentinel2:split=TEST:root=/NAS6/Members/linchenxi/projects/RS_foundation_model/satlas/clusters:extra=/NAS6/Members/linchenxi/ILSVRC"
    --local-rank=LOCAL_PROCESS_RANK
