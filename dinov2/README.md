# DINOv2 for Multimodal Remote Sensing Foundation Model Per-training
<br></br>  
## <span style=color:#4DBBD5;font-size:15px;font-weight:bold>Code Structure</span>  
```text
📦bash
 ┣ 📜submit2slurm.sh                    # submit the job to slurm
 ┣ 📜train.sh                           # the bash file to run code at server with distributed training
 ┗ 📜train_slurm.sh                     # the inner bash file called by "submit2slurm.sh"
📦configs 
┣ 📂eval                                # everything under this folder is unchanged
 ┃ ┣ ...    
 ┣ 📂train
 ┃ ┣ 📜vitg14.yaml                      # unchanged
 ┃ ┣ 📜vitl14.yaml                      # unchanged
 ┃ ┣ 📜vitl16_short.yaml                # unchanged
 ┃ ┣ 📜vitl16_short_multimodal.yaml     # the config for multimodal pretraining (Sen12MS)
 ┃ ┗ 📜vitl16_short_s2.yaml             # the config for unimodal pretraining (Satlas)
 ┗ 📜ssl_default_config.yaml            # unchanged, the default config for models
📦data
┣ 📂datasets
┃ ┣ 📜__init__.py                       # unchanged
┃ ┣ 📜decoders.py                       # unchanged
┃ ┣ 📜extended.py                       # unchanged
┃ ┣ 📜image_net.py                      # unchanged
┃ ┣ 📜image_net_22k.py                  # unchanged
┃ ┣ 📜sen12ms.py                        # the dataset class for Sen12MS-CR-TS 
┃ ┣ 📜sentinel2.py                      # the dataset class for Satlas
┃ ┗ 📜sentinel2_segmentation.py         # the children dataset class for Satlas focusing on segmentation
┣ 📜__init__.py
┣ 📜adapters.py                         # unchanged
┣ 📜augmentations.py                    # unchanged
┣ 📜augmentations_ms12.py               # the augmentation class for Sen12MS-CR-TS
┣ 📜augmentations_satlas.py             # the augmentation class for Satlas
┣ 📜collate.py                          # the collate function
┣ 📜loaders.py                          # whenever having a new dataset, add it to this file
┣ 📜masking.py                          # unchanged
┣ 📜samplers.py                         # unchanged
┗ 📜transforms.py                       # added some remote sensing-related functions
📦distributed                           # unchanged
┗ 📜__init__.py
📦eval
┣ 📂depth                               # unchanged
┣ 📂segmentation                        # unchanged
┣ 📂segmentation_m2f                    # unchanged
┣ 📜__init__.py
┣ 📜knn.py                              # unchanged
┣ 📜linear.py                           # unchanged
┣ 📜load_pretrained_model.py            # unchanged
┣ 📜log_regression.py                   # unchanged
┣ 📜metrics.py                          # unchanged
┣ 📜segmentation.py                     # unchanged
┣ 📜setup.py                            # unchanged
┣ 📜utils.py                            # unchanged
┗ 📜visualize_attention.py              # visualize the attention map of any input
📦fsdp                                  # unchanged
📦hub                                   # unchanged
📦layers                                # unchanged
📦logging_dinov2                        # unchanged
📦loss                                  # unchanged
📦models
 ┣ 📜__init__.py
 ┣ 📜multimodal.py                      # the multimodal model fusing S1 and S2
 ┗ 📜vision_transformer.py              # the original ViT model, added temporal embedding
📦run                                   # unchanged
📦train
 ┣ 📜__init__.py
 ┣ 📜ssl_meta_arch.py                   # the main framework of DINOv2
 ┣ 📜train.py                           # the entrance to train the unimodal DINOv2
 ┗ 📜train_multimodal.py                # the entrance to train the mutlimodal DINOv2
📦utils                                 # unchanged                     
```
--- 
## <span style=color:#4DBBD5;font-size:15px;font-weight:bold>Start the training</span>  
<span style=font-size:13px;color:#00A087>

To start the pre-training at servers, e.g., 185, run `/dinov2/dinov2/bash/train.sh`
It looks like the following but one can add more arguments, see `/dinov2/train/train_multimodal.py` for more details
```python
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
```
Similarly, run `/dinov2/dinov2/bash/sumbit2slurm.sh` at server 215 to start the pre-trainig on slurm.
</span>

---

## <span style=color:#4DBBD5;font-size:15px;font-weight:bold>Config files</span>  
<span style=font-size:13px;color:#00A087>

The default config is stored in `/dinov2/configs/ssl_default_config.yaml`, but one can customize their own configs, e.g., `/dinov2/configs/train/vitl16_short_multimodal.yaml`.

Some important arguments are listed below:
```python
train:
  dataset_path: Multimodality:split=TRAIN:root=/NAS3/Members/linchenxi/projects/foundation_model/sen12ms    # location of the dataset, "Multimodality" denotes the type of the dataset, different types can be found in /dinov2/data/loaders.py
  batch_size_per_gpu: 64
  num_workers: 5
  multimodal: true  # whether to perform multimodal pre-training, if true, the customized mutlimodal model will be initialized, otherwise, the default ViT for unimodal
student:
  arch: vit_base  # the architecture used to initialize the ViT, only work for unimodal pre-training
  arch_s1: vit_base  # the architecture used to initialize the ViT for s1, only work for multimodal pre-training
  arch_s2: vit_base  # the architecture used to initialize the ViT for s2, only work for multimodal pre-training
  block_chunks: 4   # not so sure about what it will be used for, but I guess the layers will be divided into chunks
  num_register_tokens: 2    # how many registers to use. Registers are inserted after the "CLS" token but before the patch tokens, please refer to the paper "Vision Transformers Need Registers"
  in_chans_s1: 2
  in_chans_s2: 13
  fuse_alg: "linear"    # how to fuse different multidalities, either be "linear" or "xattn" (cross-attention)
  xattn:    # parameters for cross-attention
    nhead: 4
    nlayer: 6
    dropout: 0
crops:  # the cropped image size for global view and local view
  global_crops_size: 192    # global views often have size larger than 50% of the image size 
  local_crops_size: 96      # local views often have size smaller than 50% of the image size 
ibot:
  mask_sample_probability: 0.3 
  mask_ratio_min_max:
  - 0.1
  - 0.3
```
</span>
