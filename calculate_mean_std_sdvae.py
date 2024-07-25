"""
We provide Tokenizer Evaluation code here.
Refer to 
https://github.com/richzhang/PerceptualSimilarity
https://github.com/mseitzer/pytorch-fid
"""
import argparse
import os
import sys
sys.path.append(os.getcwd())
import torch
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm

from evaluation_sdvae import load_vqgan_new, instantiate_from_config



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    config_file = args.config_file #the original file can be training yaml but it is not suitable for inference
    configs = OmegaConf.load(config_file)
    configs.data.init_args.validation.params.config.size = args.image_size
    configs.data.init_args.batch_size = args.batch_size

    model = load_vqgan_new(configs, args.ckpt_path).to(DEVICE) #please specify your own path here

    configs.data.init_args.pop("train", None)
    configs.data.init_args.pop("test", None)
    dataset = instantiate_from_config(configs.data)
    dataset.prepare_data()
    dataset.setup()

    latents = []
    with torch.no_grad():
        for batch in tqdm(dataset._val_dataloader()):
            images = batch["image"].permute(0, 3, 1, 2).to(DEVICE)
            quant = model.encode(images).latent_dist.mode()

            latent_tmp = torch.flatten(quant).cpu().numpy()
            latents.append(latent_tmp)

    latents = np.concatenate(latents, axis=0)
    mean = np.mean(latents)
    var = np.var(latents)
    print(f"[Info] Model: {args.config_file} Mean: {mean}, Var: {var}")

    
  
def get_args():
   parser = argparse.ArgumentParser(description="inference parameters")
   parser.add_argument("--config_file", required=True, type=str)
   parser.add_argument("--ckpt_path", required=True, type=str)
   parser.add_argument("--image_size", default=256, type=int)
   parser.add_argument("--batch_size", default=8, type=int)

   return parser.parse_args()

if __name__ == "__main__":
  args = get_args()
  main(args)