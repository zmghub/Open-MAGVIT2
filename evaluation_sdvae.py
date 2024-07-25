"""
We provide Tokenizer Evaluation code here.
Refer to 
https://github.com/richzhang/PerceptualSimilarity
https://github.com/mseitzer/pytorch-fid
"""
import argparse
import os
import sys

import torch.distributed
sys.path.append(os.getcwd())
import torch
import torchvision.transforms as transforms
import fairscale.nn.model_parallel.initialize as fs_init
from omegaconf import OmegaConf
import importlib
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers.models import AutoencoderKL
from taming.sdxl.autoencoder_kl import AutoencoderKL as AutoencoderKL_sdxl
from metrics.inception import InceptionV3
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from metrics.fid import calculate_frechet_distance

def load_vqgan_new(config, ckpt_path=None, is_gumbel=False):
  if config.model.class_path == "sd-vae-ft-ema":
     model = AutoencoderKL.from_pretrained(ckpt_path)
     return model.eval()
  elif config.model.class_path == "sdxl-ds16-vae":
     model = AutoencoderKL_sdxl.from_pretrained(ckpt_path)
     return model.eval()
  else:
    raise ValueError


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

def get_images_part(batch, rank, world_size):
  if world_size == 1:
    return batch
  b,c,h,w = batch.shape
  b_r = b // world_size
  start = rank * b_r
  if rank == world_size - 1:
    end = b
  else:
    end = start + b_r
  return batch[start:end]

def gather_tensors(x, world_size):
  # 获取形状
  local_shape = x.size()
  local_shape = torch.tensor(local_shape, device=x.device)
  shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
  torch.distributed.all_gather(shapes, local_shape)
  # 默认c h w一致，仅可能在b维度出现不同
  shapes = [i.int().cpu().numpy().tolist() for i in shapes]
  b_all = [i[0] for i in shapes]
  b_max = max(b_all)
  if b_max != local_shape[0]:
    x_diff = torch.zeros(b_max - local_shape[0], *local_shape[1:]).to(dtype=x.dtype, device=x.device)
    x = torch.cat([x, x_diff], dim=0)
  # 获取tensor
  samples = [torch.zeros_like(x) for _ in range(world_size)]
  torch.distributed.all_gather(samples, x)
  samples = [item[:b_all[i]] for i, item in enumerate(samples)]
  samples = torch.cat(samples, dim=0)
  if os.environ.get("DEBUG", "0") == "1":
    print(f"debug info: source shape: {local_shape}, gathered shape: {samples.shape}")
  return samples

def reduce_tensor(x, world_size):
  torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
  return x

def main(args):
    torch.cuda.empty_cache()
    # Setup PyTorch:
    torch.set_grad_enabled(False)

    # dist init
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    fs_init.initialize_model_parallel(1)
    torch.manual_seed(local_rank)
    print(f"WORLD_SIZE: {world_size}, RANK: {local_rank}, current_device: {device}")
    torch.distributed.barrier()

    config_file = args.config_file #the original file can be training yaml but it is not suitable for inference
    configs = OmegaConf.load(config_file)
    configs.data.init_args.validation.params.config.size = args.image_size
    configs.data.init_args.batch_size = args.batch_size * world_size
    configs.data.init_args.num_workers = max(1, configs.data.init_args.num_workers // world_size)

    model = load_vqgan_new(configs, args.ckpt_path).to(device) #please specify your own path here
    codebook_size = 1
    if args.visualize_dir is not None:
      visualize_dir = args.visualize_dir
      visualize_version = configs.trainer.logger.init_args.version
      visualize_original = os.path.join(visualize_dir, visualize_version, "original_{}".format(args.image_size))
      visualize_rec = os.path.join(visualize_dir, visualize_version, "rec_{}".format(args.image_size))
      if not os.path.exists(visualize_original) and local_rank == 0:
        os.makedirs(visualize_original, exist_ok=True)

      if not os.path.exists(visualize_rec) and local_rank == 0:
        os.makedirs(visualize_rec, exist_ok=True)
    
    #usage
    usage = {}
    for i in range(codebook_size):
      usage[i] = 0


    # FID score related
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).to(device)
    inception_model.eval()

    configs.data.init_args.pop("train", None)
    configs.data.init_args.pop("test", None)
    dataset = instantiate_from_config(configs.data)
    dataset.prepare_data()
    dataset.setup()
    pred_xs = []
    pred_recs = []

    # LPIPS score related
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)   # closer to "traditional" perceptual loss, when used for optimization
    lpips_alex = 0.0
    lpips_vgg = 0.0

    # SSIM score related
    ssim_value = 0.0

    # PSNR score related
    psnr_value = 0.0

    num_images = 0
    num_iter = 0
    custom_to_01_fn = custom_to_01_svd if os.getenv("SVD_FLAG", "false").lower() == "true" else custom_to_01
    custom_to_pil_fn = custom_to_pil_svd if os.getenv("SVD_FLAG", "false").lower() == "true" else custom_to_pil
    with torch.no_grad():
        for batch in tqdm(dataset._val_dataloader(), ncols=80, desc=f"rank-{local_rank} pid:{str(os.getpid())}", position=local_rank):
            images = batch["image"].permute(0, 3, 1, 2)
            num_images += images.shape[0]
            images_rank = get_images_part(images, local_rank, world_size).to(device)

            reconstructed_images_rank = model(images_rank).sample
            if local_rank == 0 and args.visualize_dir is not None and num_iter < 50:
              image_save = custom_to_pil_fn(images_rank[0])
              reconstructed_image_save = custom_to_pil_fn(reconstructed_images_rank[0])
              image_save.save(os.path.join(visualize_original, "{}.png".format(num_iter)))
              reconstructed_image_save.save(os.path.join(visualize_rec, "{}.png".format(num_iter)))

            images_rank = custom_to_01_fn(images_rank)
            reconstructed_images_rank = custom_to_01_fn(reconstructed_images_rank)

            # calculate lpips
            lpips_alex += loss_fn_alex(images_rank, reconstructed_images_rank, normalize=True).sum()
            lpips_vgg += loss_fn_vgg(images_rank, reconstructed_images_rank, normalize=True).sum()


            # images = (images + 1) / 2
            # reconstructed_images = (reconstructed_images + 1) / 2

            # calculate fid
            pred_x = inception_model(images_rank)[0]
            pred_x = pred_x.squeeze(3).squeeze(2).cpu().numpy()
            pred_rec = inception_model(reconstructed_images_rank)[0]
            pred_rec = pred_rec.squeeze(3).squeeze(2).cpu().numpy()

            pred_xs.append(pred_x)
            pred_recs.append(pred_rec)

            #calculate PSNR and SSIM
            rgb_restored = (reconstructed_images_rank * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            rgb_gt = (images_rank * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
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

    lpips_alex = reduce_tensor(lpips_alex, world_size)
    lpips_vgg = reduce_tensor(lpips_vgg, world_size)
    ssim_value = torch.tensor(ssim_value).to(device)
    psnr_value = torch.tensor(psnr_value).to(device)
    ssim_value = reduce_tensor(ssim_value, world_size).item() / world_size
    psnr_value = reduce_tensor(psnr_value, world_size).item() / world_size
    pred_xs = torch.tensor(np.concatenate(pred_xs, axis=0)).to(device)
    pred_recs = torch.tensor(np.concatenate(pred_recs, axis=0)).to(device)
    pred_xs = gather_tensors(pred_xs, world_size).cpu().numpy()
    pred_recs = gather_tensors(pred_recs, world_size).cpu().numpy()
    if local_rank == 0:
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
   parser.add_argument("--visualize_dir", type=str, default=None)

   return parser.parse_args()

if __name__ == "__main__":
  args = get_args()
  main(args)