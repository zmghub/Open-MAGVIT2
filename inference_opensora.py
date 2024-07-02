"""
We provide Tokenizer Inference code here.
"""

import os
import sys
sys.path.append(os.getcwd())
import torch
from omegaconf import OmegaConf
import importlib
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from taming.models.lfqgan import VQModel
from taming.models.klgan import KLModel
from taming.svd.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from taming.svd.autoencoder_vq_temporal_decoder import AutoencoderVQTemporalDecoder
import argparse

from evaluation_opensora import load_vqgan_new, instantiate_from_config, custom_to_pil, custom_to_pil_svd

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
   config_file = args.config_file #the original file can be training yaml but it is not suitable for inference
   configs = OmegaConf.load(config_file)
   configs.data.init_args.batch_size = 1 # change the batch size
   configs.data.init_args.test.params.config.size = args.image_size #using test to inference

   model = load_vqgan_new(configs, args.ckpt_path).to(DEVICE)

   visualize_dir = args.visualize_dir
   visualize_version = configs.trainer.logger.init_args.version
   visualize_original = os.path.join(visualize_dir, visualize_version, "original_{}".format(args.image_size))
   visualize_rec = os.path.join(visualize_dir, visualize_version, "rec_{}".format(args.image_size))
   if not os.path.exists(visualize_original):
      os.makedirs(visualize_original, exist_ok=True)
   
   if not os.path.exists(visualize_rec):
      os.makedirs(visualize_rec, exist_ok=True)
   
   configs.data.init_args.pop("train", None)
   configs.data.init_args.pop("validation", None)
   dataset = instantiate_from_config(configs.data)
   dataset.prepare_data()
   dataset.setup()

   count = 0
   if args.image_num == -1:
      args.image_num = len(dataset._test_dataloader())

   custom_to_pil_fn = custom_to_pil_svd if os.getenv("SVD_FLAG", "false").lower() == "true" else custom_to_pil
   with torch.no_grad():
      for idx, batch in tqdm(enumerate(dataset._test_dataloader())):
         if count > args.image_num:
            break
         images = batch["image"].permute(0, 3, 1, 2).to(DEVICE)

         count += images.shape[0]
         reconstructed_images, _, _, indices, _ = model(images)
                  
         
         image = images[0]
         reconstructed_image = reconstructed_images[0]

         image = custom_to_pil_fn(image)
         reconstructed_image = custom_to_pil_fn(reconstructed_image)

         image.save(os.path.join(visualize_original, "{}.png".format(idx)))
         reconstructed_image.save(os.path.join(visualize_rec, "{}.png".format(idx)))

    
def get_args():
   parser = argparse.ArgumentParser(description="inference parameters")
   parser.add_argument("--config_file", required=True, type=str)
   parser.add_argument("--ckpt_path", required=True, type=str)
   parser.add_argument("--image_size", default=256, type=int)
   parser.add_argument("--image_num", default=50, type=int)
   parser.add_argument("--visualize_dir", type=str, default="./logs/visualize")

   return parser.parse_args()
  
if __name__ == "__main__":
   args = get_args()
   main(args)