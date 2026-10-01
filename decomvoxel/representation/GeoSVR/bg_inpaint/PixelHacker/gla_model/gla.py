import warnings
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.cache_utils import Cache, DynamicCache
from diffusers.utils import logging

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

# torch<=2.4 compatibility: fla expects symbols on torch.distributed.tensor,
# while this runtime exposes them under torch.distributed._tensor.
try:
    import torch.distributed.tensor as _td_tensor
    from torch.distributed import _tensor as _td_legacy
    for _name in dir(_td_legacy):
        if not hasattr(_td_tensor, _name):
            try:
                setattr(_td_tensor, _name, getattr(_td_legacy, _name))
            except Exception:
                pass
except Exception:
    pass

# fla registers many configs/models into transformers auto registries.
# In newer transformers versions, some keys can already exist; during fla import
# treat duplicates as exist_ok.
_orig_cfg_register = None
_orig_lazy_register = None
try:
    from transformers.models.auto import configuration_auto as _cfg_auto
    from transformers.models.auto import auto_factory as _auto_factory

    _orig_cfg_register = _cfg_auto.CONFIG_MAPPING.register
    _orig_lazy_register = _auto_factory._LazyAutoMapping.register

    def _cfg_register_compat(key, value, exist_ok=False):
        try:
            return _orig_cfg_register(key, value, exist_ok=exist_ok)
        except ValueError as e:
            msg = str(e)
            if "already used by a Transformers config" in msg:
                return _orig_cfg_register(key, value, exist_ok=True)
            raise

    def _lazy_register_compat(self, key, value, exist_ok=False):
        try:
            return _orig_lazy_register(self, key, value, exist_ok=exist_ok)
        except ValueError as e:
            msg = str(e)
            if "already used by a Transformers model" in msg:
                return _orig_lazy_register(self, key, value, exist_ok=True)
            raise

    _cfg_auto.CONFIG_MAPPING.register = _cfg_register_compat
    _auto_factory._LazyAutoMapping.register = _lazy_register_compat
except Exception:
    _orig_cfg_register = None
    _orig_lazy_register = None

from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from fla.models.gla.modeling_gla import GLAMLP, GLABlock, GatedLinearAttention, GLAConfig
from fla.modules import FusedCrossEntropyLoss, RMSNorm, ShortConvolution, FusedRMSNormSwishGate

if _orig_cfg_register is not None:
    try:
        _cfg_auto.CONFIG_MAPPING.register = _orig_cfg_register
    except Exception:
        pass
if _orig_lazy_register is not None:
    try:
        _auto_factory._LazyAutoMapping.register = _orig_lazy_register
    except Exception:
        pass

import torch._dynamo
torch._dynamo.config.suppress_errors = True

logger = logging.get_logger(__name__)


class GatedLinearAttention_ForwardWrapper(GatedLinearAttention):
    def forward(self,*args,**kwargs):
        return super(GatedLinearAttention_ForwardWrapper, self).forward(*args,**kwargs)[0]


DEFAULT_GLA_CONFIG = dict(
    mode = 'chunk',
    hidden_size = 2048,
    expand_k = 0.5,
    expand_v = 1.0,
    num_heads = 4,
    num_kv_heads = None,
    feature_map = None,
    use_short_conv = False,
    conv_size = 4,
    conv_bias = False,
    use_output_gate = True,
    gate_fn = 'swish',
    elementwise_affine = True,
    norm_eps = 1e-6,
    gate_logit_normalizer = 16,
    gate_low_rank_dim = 16,
    clamp_min = None,
    fuse_norm = True,
    layer_idx = None,
)


