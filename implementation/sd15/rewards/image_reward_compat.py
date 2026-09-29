"""SatisDive sd15: image reward compat. See README.md for the supported interface."""
from __future__ import annotations

import sys
from typing import Any

import torch  # noqa: F401  -- transformers needs torch loaded


def _install_stubs() -> None:
    """Inject stubs for symbols ImageReward expects in transformers.modeling_utils."""
    import transformers.modeling_utils as mu

    if not hasattr(mu, "find_pruneable_heads_and_indices"):
        def _find_pruneable_heads_and_indices(*_args: Any, **_kwargs: Any):
            raise RuntimeError(
                "find_pruneable_heads_and_indices is a stub: head pruning is "
                "not supported in this compatibility shim. ImageReward scoring "
                "does not use this codepath."
            )
        mu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

    if not hasattr(mu, "prune_linear_layer"):
        def _prune_linear_layer(*_args: Any, **_kwargs: Any):
            raise RuntimeError(
                "prune_linear_layer is a stub: layer pruning is not supported "
                "in this compatibility shim."
            )
        mu.prune_linear_layer = _prune_linear_layer


def _patch_blip_init_tokenizer() -> None:
    """Replace blip.init_tokenizer with a transformers-5.x-compatible version.

    The original uses ``tokenizer.additional_special_tokens_ids[0]`` which was
    removed in transformers 5.x. Replacement uses the explicit token name.

    blip_pretrain.py imports ``init_tokenizer`` at module-load time
    (``from .blip import init_tokenizer``), so we have to update BOTH the
    source binding (``blip.init_tokenizer``) AND the captured reference in
    ``blip_pretrain``.
    """
    from ImageReward.models.BLIP import blip as _blip
    from ImageReward.models.BLIP import blip_pretrain as _blip_pretrain
    from transformers import BertTokenizer

    def _init_tokenizer():
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        # Compat: transformers 5.x removed .additional_special_tokens_ids;
        # look up the token id we just added by name.
        tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
        return tokenizer

    _blip.init_tokenizer = _init_tokenizer
    _blip_pretrain.init_tokenizer = _init_tokenizer


def load_image_reward(name: str = "ImageReward-v1.0"):
    """Load an ImageReward model with the compatibility shim applied.

    Args:
        name: Model identifier (default 'ImageReward-v1.0').

    Returns:
        The loaded ImageReward model. Use ``model.score(prompt, image)`` to
        score a single (prompt, PIL.Image) pair.
    """
    _install_stubs()

    # Bypass ``ImageReward/__init__.py`` because it does ``from .ReFL import *``
    # which pulls in ``wandb``. Our wandb install is broken on this venv, but
    # ReFL is fine-tuning code we never need for inference. Stub the package
    # to avoid the parent __init__ running, then import the utilities module
    # which exposes ``load`` (the only function we need).
    if "ImageReward" not in sys.modules:
        import importlib.util
        import types
        spec = importlib.util.find_spec("ImageReward")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("ImageReward package not found on sys.path")
        pkg = types.ModuleType("ImageReward")
        pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules["ImageReward"] = pkg

    # Importing utils pulls in models.BLIP.blip transitively, which defines
    # init_tokenizer. We then monkey-patch it before any actual model load,
    # so the tokenizer initialization uses transformers-5.x-compatible API.
    from ImageReward import utils as ir_utils  # noqa: F401
    _patch_blip_init_tokenizer()
    return ir_utils.load(name)


class IRScorer:
    """Wrapper exposing the same .score(images, prompt) API as HPSScorer."""

    def __init__(self, model=None):
        if model is None:
            model = load_image_reward()
        self.model = model

    def score(self, images, prompt: str):
        scores = []
        for img in images:
            s = self.model.score(prompt, img)
            scores.append(float(s))
        return torch.tensor(scores)
