"""Safetensors sidecar store for MiniMax-H3 layer-50 prompt embeddings."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from unirl.config.require import require
from unirl.types.conditions import TextEmbedCondition

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.json"
SCHEMA_VERSION = "1.0"
EXTRACTOR = "minimax_h3_text_embed"
FEATURE_DIM = 5120
TARGET_LAYER = 50


def compute_prompt_key(prompt: str) -> str:
    """Return a 16-hex SHA-256 fingerprint of the stripped prompt."""
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()[:16]


class OfflineTextEmbedStore:
    """Read-only mmap store: ``index.json`` plus sharded ``.safetensors`` files."""

    def __init__(self, cache_dir: str, entries: Dict[str, str]) -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        self.entries = entries
        # mmap handles, not materialized shards: a shard holds ~2000 [L, 5120]
        # tensors, and every DP rank on a node would otherwise keep its own copy.
        self._shard_handles: Dict[str, Any] = {}

    @classmethod
    def from_dir(cls, cache_dir: str) -> "OfflineTextEmbedStore":
        """Load ``index.json`` from ``cache_dir``."""
        index_path = os.path.join(cache_dir, INDEX_FILENAME)
        require(os.path.isdir(cache_dir), f"OfflineTextEmbedStore: cache directory not found: {cache_dir}")
        require(
            os.path.isfile(index_path),
            f"OfflineTextEmbedStore: index file not found: {index_path}. "
            "Run `python -m unirl.tools.precompute_minimax_h3` first.",
        )
        with open(index_path, "r", encoding="utf-8") as handle:
            index_data = json.load(handle)
        extractor = index_data.get("extractor")
        require(
            extractor == EXTRACTOR,
            f"OfflineTextEmbedStore: extractor {extractor!r} != {EXTRACTOR!r} at {cache_dir}",
        )
        entries = {key: str(shard) for key, shard in index_data["entries"].items()}
        logger.info("Loaded MiniMax-H3 text-embed store from %s (%d prompts)", cache_dir, len(entries))
        return cls(cache_dir, entries)

    def contains(self, prompt: str) -> bool:
        return compute_prompt_key(prompt) in self.entries

    def _open_shard(self, shard_name: str) -> Any:
        if shard_name not in self._shard_handles:
            shard_path = os.path.join(self.cache_dir, shard_name)
            require(os.path.isfile(shard_path), f"OfflineTextEmbedStore: missing shard: {shard_path}")
            self._shard_handles[shard_name] = safe_open(shard_path, framework="pt", device="cpu")
        return self._shard_handles[shard_name]

    def get(self, prompt: str) -> TextEmbedCondition:
        """Fetch one prompt's unpadded embedding as batch-1 ``[1, L, D]``."""
        key = compute_prompt_key(prompt)
        require(
            key in self.entries,
            f"OfflineTextEmbedStore: prompt not in cache: {prompt!r}. "
            "Run `python -m unirl.tools.precompute_minimax_h3` over every prompt file the run reads, eval included.",
        )
        shard_name = self.entries[key]
        shard = self._open_shard(shard_name)
        require(key in shard.keys(), f"OfflineTextEmbedStore: key {key} missing from shard {shard_name}")
        tensor = shard.get_tensor(key)
        require(tensor.dim() == 2, f"OfflineTextEmbedStore: expected [L, D], got {tuple(tensor.shape)}")
        embeds = tensor.unsqueeze(0)
        return TextEmbedCondition(
            embeds=embeds,
            attn_mask=torch.ones((1, embeds.shape[1]), dtype=torch.bool, device=embeds.device),
        )

    def verify_coverage_or_raise(self, prompts: Sequence[str], *, context: str = "") -> None:
        """Raise if any prompt is missing from the store."""
        missing = [prompt for prompt in prompts if not self.contains(prompt)]
        if not missing:
            return
        where = f" in {context}" if context else ""
        raise ValueError(
            f"Precomputed MiniMax-H3 text-embed coverage failed{where}: "
            f"{len(missing)}/{len(prompts)} prompts missing from {self.cache_dir}. "
            f"First missing: {missing[:3]!r}."
        )


class OfflineTextEmbedWriter:
    """Write unpadded ``[L, D]`` embeddings into safetensors shards plus ``index.json``."""

    def __init__(
        self,
        output_dir: str,
        *,
        model_checkpoint: str,
        dtype: str = "bfloat16",
        shard_size: int = 2000,
        resume: bool = False,
        force_overwrite: bool = False,
    ) -> None:
        self.output_dir = os.path.abspath(output_dir)
        self.model_checkpoint = model_checkpoint
        self.dtype = dtype
        self.shard_size = shard_size
        self.resume = resume
        self.entries: Dict[str, str] = {}
        self.shards: List[str] = []
        self._current_shard_tensors: Dict[str, torch.Tensor] = {}
        self._current_shard_idx = 0
        self._init_output_dir(resume=resume, force_overwrite=force_overwrite)

    def _init_output_dir(self, *, resume: bool, force_overwrite: bool) -> None:
        index_path = os.path.join(self.output_dir, INDEX_FILENAME)
        if os.path.exists(index_path):
            if force_overwrite:
                with open(index_path, "r", encoding="utf-8") as handle:
                    old = json.load(handle)
                for shard in old.get("shards", []):
                    shard_path = os.path.join(self.output_dir, shard)
                    if os.path.isfile(shard_path):
                        os.remove(shard_path)
                os.remove(index_path)
                logger.info("force_overwrite: cleared existing cache at %s", self.output_dir)
            elif resume:
                with open(index_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if data.get("extractor") != EXTRACTOR or int(data.get("feature_dim", 0)) != FEATURE_DIM:
                    raise ValueError(
                        f"OfflineTextEmbedWriter: fingerprint mismatch at {index_path} "
                        f"(extractor={data.get('extractor')}, dim={data.get('feature_dim')}). Use --force-overwrite."
                    )
                self.entries = {key: str(shard) for key, shard in data.get("entries", {}).items()}
                self.shards = list(data.get("shards", []))
                self._current_shard_idx = len(self.shards)
                logger.info(
                    "Resumed cache at %s (%d entries, %d shards)", self.output_dir, len(self.entries), len(self.shards)
                )
            else:
                raise FileExistsError(
                    f"OfflineTextEmbedWriter: {index_path} already exists. Pass --resume or --force-overwrite."
                )
        os.makedirs(self.output_dir, exist_ok=True)

    def contains(self, prompt: str) -> bool:
        return compute_prompt_key(prompt) in self.entries

    def add(self, prompt: str, tensor: torch.Tensor) -> None:
        """Append one unpadded ``[L, D]`` (or ``[1, L, D]``) embedding."""
        key = compute_prompt_key(prompt)
        if self.resume and key in self.entries:
            return
        value = tensor.detach().to("cpu").contiguous()
        if value.dim() == 3 and value.shape[0] == 1:
            value = value.squeeze(0)
        require(value.dim() == 2, f"OfflineTextEmbedWriter: expected [L, D], got {tuple(value.shape)}")
        require(
            int(value.shape[1]) == FEATURE_DIM,
            f"OfflineTextEmbedWriter: expected dim={FEATURE_DIM}, got {value.shape[1]}",
        )
        shard_name = f"embeddings_{self._current_shard_idx:05d}.safetensors"
        self._current_shard_tensors[key] = value
        self.entries[key] = shard_name
        if len(self._current_shard_tensors) >= self.shard_size:
            self._flush_current_shard()

    def _flush_current_shard(self) -> None:
        if not self._current_shard_tensors:
            return
        shard_name = f"embeddings_{self._current_shard_idx:05d}.safetensors"
        shard_path = os.path.join(self.output_dir, shard_name)
        logger.info("Saving %d embeddings to %s", len(self._current_shard_tensors), shard_path)
        save_file(self._current_shard_tensors, shard_path)
        if shard_name not in self.shards:
            self.shards.append(shard_name)
        self._current_shard_tensors.clear()
        self._current_shard_idx += 1
        self._save_index()

    def _save_index(self) -> None:
        index_data = {
            "schema_version": SCHEMA_VERSION,
            "extractor": EXTRACTOR,
            "model_checkpoint": self.model_checkpoint,
            "target_layer": TARGET_LAYER,
            "feature_dim": FEATURE_DIM,
            "dtype": self.dtype,
            "total_prompts": len(self.entries),
            "shards": self.shards,
            "entries": self.entries,
        }
        index_path = os.path.join(self.output_dir, INDEX_FILENAME)
        tmp_path = index_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(index_data, handle, indent=2, ensure_ascii=False)
        os.replace(tmp_path, index_path)

    def close(self) -> None:
        """Flush the open shard and write the final index."""
        if self._current_shard_tensors:
            self._flush_current_shard()
        else:
            self._save_index()
        logger.info("Wrote MiniMax-H3 text-embed store to %s (%d prompts)", self.output_dir, len(self.entries))


__all__ = [
    "EXTRACTOR",
    "FEATURE_DIM",
    "INDEX_FILENAME",
    "TARGET_LAYER",
    "OfflineTextEmbedStore",
    "OfflineTextEmbedWriter",
    "compute_prompt_key",
]
