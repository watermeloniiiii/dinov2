from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="/NAS6/Members/linchenxi/ILSVRC", extra="/NAS6/Members/linchenxi/ILSVRC")
    dataset.dump_extra()
