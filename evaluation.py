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
import torchvision.transforms as transforms
from omegaconf import OmegaConf
import importlib
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import linalg

from taming.models.lfqgan import VQModel
from taming.models.klgan import KLModel
from taming.svd.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from taming.svd.autoencoder_vq_temporal_decoder import AutoencoderVQTemporalDecoder
from metrics.inception import InceptionV3
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from metrics.fid import calculate_frechet_distance

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan_new(config, ckpt_path=None, is_gumbel=False):
  if config.model.class_path.endswith("VQModel"):
    model = VQModel(**config.model.init_args)
  elif config.model.class_path.endswith("KLModel"):
    model = KLModel(**config.model.init_args)
  elif config.model.class_path.endswith("AutoencoderKLTemporalDecoder"):
    model = AutoencoderKLTemporalDecoder(**config.model.init_args)
  elif config.model.class_path.endswith("AutoencoderVQTemporalDecoder"):
    model = AutoencoderVQTemporalDecoder(**config.model.init_args)
  else:
    raise ValueError
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")
    if 'state_dict' in sd:
      sd = sd["state_dict"]
    if 'model' in sd:
      sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=True)
    print(f"[Info] Load model weight from {ckpt_path}")
    print(f"[Info] Missing keys: {missing}")
    print(f"[Info] Unexpected keys: {unexpected}")
  return model.eval()


def get_obj_from_str(string, reload=False):
    print(string)
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "class_path" in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def custom_to_pil_svd(x):
  x = x.detach().cpu()
  x = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])(x)
  x = (np.clip(x.numpy(), 0,1) * 255).astype('uint8')
  x = x.transpose((1, 2, 0))
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def custom_to_01_svd(x):
  x = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])(x)
  return x.clamp(0, 1)

def custom_to_01(x):
   x = (x + 1) / 2
   return x

def main(args):
    config_file = args.config_file #the original file can be training yaml but it is not suitable for inference
    configs = OmegaConf.load(config_file)
    configs.data.init_args.validation.params.config.size = args.image_size
    configs.data.init_args.batch_size = args.batch_size

    model = load_vqgan_new(configs, args.ckpt_path).to(DEVICE) #please specify your own path here
    if isinstance(model, (VQModel, AutoencoderVQTemporalDecoder)):
      codebook_size = configs.model.init_args.n_embed
    elif isinstance(model, (KLModel, AutoencoderKLTemporalDecoder)):
      codebook_size = 1
    
    #usage
    usage = {}
    for i in range(codebook_size):
      usage[i] = 0


    # FID score related
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).to(DEVICE)
    inception_model.eval()

    configs.data.init_args.pop("train", None)
    configs.data.init_args.pop("test", None)
    dataset = instantiate_from_config(configs.data)
    dataset.prepare_data()
    dataset.setup()
    pred_xs = []
    pred_recs = []

    # LPIPS score related
    loss_fn_alex = lpips.LPIPS(net='alex').to(DEVICE)  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(DEVICE)   # closer to "traditional" perceptual loss, when used for optimization
    lpips_alex = 0.0
    lpips_vgg = 0.0

    # SSIM score related
    ssim_value = 0.0

    # PSNR score related
    psnr_value = 0.0

    num_images = 0
    num_iter = 0
    custom_to_01_fn = custom_to_01_svd if os.getenv("SVD_FLAG", "false").lower() == "true" else custom_to_01
    with torch.no_grad():
        for batch in tqdm(dataset._val_dataloader()):
            images = batch["image"].permute(0, 3, 1, 2).to(DEVICE)
            num_images += images.shape[0]

            if isinstance(model, (VQModel, AutoencoderVQTemporalDecoder)):
              if model.use_ema:
                  with model.ema_scope():
                      quant, diff, indices, _ = model.encode(images)
                      reconstructed_images = model.decode(quant)
              else:
                quant, diff, indices, _ = model.encode(images)
                reconstructed_images = model.decode(quant)
              ### usage
              for index in indices:
                usage[index.item()] += 1

            elif isinstance(model, (KLModel, AutoencoderKLTemporalDecoder)):
              if model.use_ema:
                  with model.ema_scope():
                      quant, diff = model.encode(images)
                      reconstructed_images = model.decode(quant)
              else:
                quant, diff = model.encode(images)
                reconstructed_images = model.decode(quant)

            images = custom_to_01_fn(images)
            reconstructed_images = custom_to_01_fn(reconstructed_images)

            # calculate lpips
            lpips_alex += loss_fn_alex(images, reconstructed_images, normalize=True).sum()
            lpips_vgg += loss_fn_vgg(images, reconstructed_images, normalize=True).sum()


            # images = (images + 1) / 2
            # reconstructed_images = (reconstructed_images + 1) / 2

            # calculate fid
            pred_x = inception_model(images)[0]
            pred_x = pred_x.squeeze(3).squeeze(2).cpu().numpy()
            pred_rec = inception_model(reconstructed_images)[0]
            pred_rec = pred_rec.squeeze(3).squeeze(2).cpu().numpy()

            pred_xs.append(pred_x)
            pred_recs.append(pred_rec)

            #calculate PSNR and SSIM
            rgb_restored = (reconstructed_images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_gt = (images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_restored = rgb_restored.astype(np.float32) / 255.
            rgb_gt = rgb_gt.astype(np.float32) / 255.
            ssim_temp = 0
            psnr_temp = 0
            B, _, _, _ = rgb_restored.shape
            for i in range(B):
                rgb_restored_s, rgb_gt_s = rgb_restored[i], rgb_gt[i]
                ssim_temp += ssim_loss(rgb_restored_s, rgb_gt_s, data_range=1.0, channel_axis=-1)
                psnr_temp += psnr_loss(rgb_gt, rgb_restored)
            ssim_value += ssim_temp / B
            psnr_value += psnr_temp / B
            num_iter += 1

    pred_xs = np.concatenate(pred_xs, axis=0)
    pred_recs = np.concatenate(pred_recs, axis=0)

    mu_x = np.mean(pred_xs, axis=0)
    sigma_x = np.cov(pred_xs, rowvar=False)
    mu_rec = np.mean(pred_recs, axis=0)
    sigma_rec = np.cov(pred_recs, rowvar=False)


    fid_value = calculate_frechet_distance(mu_x, sigma_x, mu_rec, sigma_rec)
    lpips_alex_value = lpips_alex / num_images
    lpips_vgg_value = lpips_vgg / num_images
    ssim_value = ssim_value / num_iter
    psnr_value = psnr_value / num_iter

    num_count = sum([1 for key, value in usage.items() if value > 0])
    utilization = num_count / codebook_size

    print("FID: ", fid_value)
    print("LPIPS_ALEX: ", lpips_alex_value.item())
    print("LPIPS_VGG: ", lpips_vgg_value.item())
    print("SSIM: ", ssim_value)
    print("PSNR: ", psnr_value)
    print("utilization", utilization)
  
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