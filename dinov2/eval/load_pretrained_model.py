from dinov2.eval.setup import setup_and_build_model
import argparse
import logging

if __name__ == "__main__":
    description = "Load DINOv2 Pretrained Model"
    args_parser = argparse.ArgumentParser()
    args = args_parser.parse_args()
    args.config_file = "/NAS6/Members/linchenxi/projects/DINOV2/model8/config.yaml"
    args.pretrained_weights = "/NAS6/Members/linchenxi/projects/DINOV2/model8/eval/training_24999/teacher_checkpoint.pth"
    args.output_dir = ""
    args.opts = []
    model, autocast_dtype = setup_and_build_model(args)
    logger = logging.getLogger()
    logger.info("Success!")
    