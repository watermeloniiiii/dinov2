# DINOv2 for Multimodal Remote Sensing Foundation Model Per-training
## <span style=color:#4DBBD5;font-size:15px;font-weight:bold>Overview</span>
The repo is modified on top of the Meta DINOv2 repo by adding multimodal modules. The original DINOv2 uses a single vision transformer to process unimodal data whereas we added a fusion module to integrate both S1 and S2 imagery either through linear fusion or cross-attention.  

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
    ┣ 📂deprecated
      ┃ ┣ 📜sentinel2.py                      # the dataset class for Satlas
      ┃ ┗ 📜sentinel2_segmentation.py         # the children dataset class for Satlas focusing on segmentation
    ┃ ┣ 📜__init__.py                       # unchanged
    ┃ ┣ 📜decoders.py                       # unchanged
    ┃ ┣ 📜extended.py                       # unchanged
    ┃ ┣ 📜image_net.py                      # unchanged
    ┃ ┣ 📜image_net_22k.py                  # unchanged
    ┃ ┣ 📜sen12ms.py                        # the dataset class for Sen12MS-CR-TS 
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
📦utils                                  # unchanged                     
```
--- 
## <span style=color:#4DBBD5;font-size:15px;font-weight:bold>Start the training</span>  
<span style=font-size:13px;color:#00A087>

To start the pre-training at servers, e.g., 185, run `/dinov2/dinov2/bash/train.sh`
It looks like the following but one can add more arguments, see `/dinov2/train/train_multimodal.py` for more details
```python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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

---

## <span style=color:#4DBBD5;font-size:15px;font-weight:bold>Sen12MS Dataset</span>  
<span style=font-size:13px;color:#00A087>

As the originally downloaded **Sen12MS-CR-TS** dataset is stored in NAS and does not follow a DL-friendly format (e.g,. train/vali/test sets), the `split_dataset` will find correlated S1 and S2 tile and save them into the target folder.The second step is to store all sample paths into a JSON file to make efficient retrieval. 

```python
    # Step 1. transfer the data from NAS to the target folder and split into training and validation sets
    split_dataset(
        "/NAS/datasets/PUBLIC_DATASETS/SEN12MS-CR-TS/",
        "/NAS3/Members/linchenxi/projects/foundation_model/sen12ms",
    )

    # Step 2. Store all samples into json files
    for dataset in ["train", "val"]:
        dump_entries(
            f"/NAS3/Members/linchenxi/projects/foundation_model/sen12ms/{dataset}/s1",
            f"/NAS3/Members/linchenxi/projects/foundation_model/sen12ms/{dataset}_all.json",
        )
```
The JSON file will look like:
```text
  [
    [
        "/NAS3/Members/linchenxi/projects/foundation_model/sen12ms/val/s1/s1_ROIs2017_32_ImgNo_4_2018-02-28_patch_12.tif",
        "/NAS3/Members/linchenxi/projects/foundation_model/sen12ms/val/s2/s2_ROIs2017_32_ImgNo_4_2018-02-24_patch_12.tif"
    ],
    [
        "/NAS3/Members/linchenxi/projects/foundation_model/sen12ms/val/s1/s1_ROIs2017_32_ImgNo_4_2018-02-28_patch_11.tif",
        "/NAS3/Members/linchenxi/projects/foundation_model/sen12ms/val/s2/s2_ROIs2017_32_ImgNo_4_2018-02-24_patch_11.tif"
    ],
    ...
  ]
```
</span>

## <span style=color:#4DBBD5;font-size:15px;font-weight:bold>Multimodal Fusion</span>
<span style=font-size:13px;color:#00A087>
We have provided two approaches to fuse the multimodal data.

1. **Linear fusion**. The linear fusion module contains two linear projections to first project the concatanated features to a higher dimension (by default four times of the embedding dimension) and then to the original embedding dimension.

```python
self.linear_fuse = nn.Sequential(
      nn.Linear(sum(self.embed_dim), 2 * sum(self.embed_dim)),
      nn.LayerNorm(2 * sum(self.embed_dim)),
      nn.GELU(),
      nn.Linear(2 * sum(self.embed_dim), self.embed_dim[0]),
  )
if self.fuse_alg == "linear":
      out["x_norm"] = self.linear_fuse(torch.cat([s1_out["x_norm"], s2_out["x_norm"]], dim=-1))
      out["x_norm_clstoken"] = out["x_norm"][:, 0]
      out["x_norm_patchtokens"] = out["x_norm"][:, self.args.num_register_tokens + 1 :]
```

2. **Cross-attention**. In the cross-attention, we use feature from S1 as the query and feature from S2 as the key and value, but one can switch the two. 

```python
def cross_attention(self, q, k, v):
    res1 = q
    res2 = self.attn(query=q, key=k, value=v)[0]
    res1 = res1 + self.dropout1(res2)
    res1 = self.norm1(res1)
    res2 = self.linear2(self.dropout(self.activation(self.linear1(res1))))
    res1 = res1 + self.dropout3(res2)
    res1 = self.norm3(res1)
    return res1
  
res = s1_out["x_norm"]
for _ in range(self.nlayer):
    q = res
    res = self.cross_attention(q, s2_out["x_norm"], s2_out["x_norm"])
out["x_norm_clstoken"] = res[:, 0]
out["x_norm_patchtokens"] = res[:, self.args.num_register_tokens + 1 :]
```
</span>  