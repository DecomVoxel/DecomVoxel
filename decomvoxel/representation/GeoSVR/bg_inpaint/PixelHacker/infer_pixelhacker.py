import os
import shutil
import sys
sys.path.append(".")
sys.path.append(os.getcwd())

from pathlib import Path
from tqdm import tqdm
import torch

from diffusers import DDIMScheduler
from utils import load_cfg, build_model, build_vae
from dataset import SimpleInferDataset
from pipeline import PixelHacker_Pipeline

import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="decomvoxel/representation/GeoSVR/bg_inpaint/PixelHacker/config/PixelHacker_sdvae_f8d4.yaml")
    parser.add_argument("--weight", default="decomvoxel/representation/GeoSVR/bg_inpaint/PixelHacker/weight/ft_places2/diffusion_pytorch_model.bin")
    parser.add_argument("--image_dir", default="imgs")
    parser.add_argument("--mask_dir", default="masks")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--retry_times", default=1, type=int)   # retry times, default is 1, means no retry
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    model_cfg = load_cfg(args.config)

    model = build_model(model_cfg, 20).to(device)
    state_dict = torch.load(args.weight, map_location=device, weights_only=True)
    print(model.load_state_dict(state_dict))

    vae = build_vae(model_cfg).to(device)
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        num_train_timesteps=1000, clip_sample=False)

    pipe = PixelHacker_Pipeline(
        model=model,
        vae=vae,
        scheduler=scheduler,
        device=device,
        dtype=torch.float)

    vae_ds_ratio = 2 ** (len(vae.config.block_out_channels) - 1)
    img_size = model.diff_model.config.sample_size * vae_ds_ratio
    assert img_size == model_cfg['data']['image_size']

    dataset = SimpleInferDataset(img_dir = args.image_dir, mask_dir = args.mask_dir)

    save_root = Path(args.output_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    for idx, (image, mask, iname) in tqdm(enumerate(dataset)):
        image = image.resize((img_size,img_size))
        mask = mask.resize((img_size,img_size))

        for i in range(args.retry_times):

            out = pipe(
                image, mask,
                image_size=img_size,
                num_steps=20,
                strength=0.999,
                guidance_scale=4.5,
                noise_offset=0.0357,
                paste=False,
                retry=i,                    # Uses random seed if retry > 0; default to 0 to align with training configuration for best results.
            )[0]

            W, H = dataset.img_origin_size
            out = out.resize((W, H))
            save_path = save_root.joinpath(iname.replace(".png", f"_{i}.png"))
            out.save(save_path)

        # default use retry = 0 as choosed inpainting method
        default_save_path = save_root.joinpath(iname.replace(".png", "_0.png"))
        shutil.copy(default_save_path, save_root.joinpath(iname))

