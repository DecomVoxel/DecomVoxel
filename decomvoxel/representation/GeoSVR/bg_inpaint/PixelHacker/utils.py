from typing import Union, Dict, List

import torch
import torch.nn as nn


def load_cfg(cfg_path: Union[str,Dict]):
    if isinstance(cfg_path, str):
        import yaml
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
    else: # from dict
        from omegaconf import OmegaConf
        config = OmegaConf.create(cfg_path)
    return config


class LCGModel(nn.Module):
    def __init__(self, diff_model, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        )
        self.diff_model = diff_model
        self.num_embeddings=num_embeddings

    def forward(
            self,
            noisy_latents,
            timesteps,
            input_ids):
        encoder_hidden_states = self.embedding_layer(input_ids)

        noise_pred = self.diff_model(noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states)
        return noise_pred

def build_model(
    config_path,
    num_embeddings=20,
    ):
    config = load_cfg(config_path)

    latent_size = config['data']['image_size'] // config['vae']['downsample_ratio']

    model_cfg = config['model']
    model_cfg.update(dict(sample_size=latent_size))

    model_type = model_cfg.pop("model_type")

    from gla_model.PixelHacker import PixelHacker
    if 'model_dir' in model_cfg:
        diff = PixelHacker.from_pretrained(
            model_cfg['model_dir'],
            local_files_only=True,
            reference_compile=False)
    else:
        diff = PixelHacker(**model_cfg)
    embedding_dim = diff.config.encoder_hid_dim

    return LCGModel(diff_model=diff,num_embeddings=num_embeddings,embedding_dim=embedding_dim)


from diffusers import DDIMScheduler, AutoencoderKL
def build_vae(model_cfg):
    if 'sdvae' in model_cfg["vae"]["model_name"]:
        vae = AutoencoderKL.from_pretrained(
            model_cfg["vae"]['model_dir'],
            local_files_only=True,
            use_safetensors=False,
            low_cpu_mem_usage=False,
        )
    return vae
