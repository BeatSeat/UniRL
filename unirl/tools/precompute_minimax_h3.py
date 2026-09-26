"""CLI: precompute MiniMax-H3 Qwen3-VL layer-50 prompt embeddings to safetensors shards."""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import List, Optional

import torch

from unirl.data.datasets import TextPromptDataset
from unirl.models.minimax_h3.offline_text_embed import OfflineTextEmbedStore, OfflineTextEmbedWriter
from unirl.models.minimax_h3.text_embed import encode_minimax_h3_prompt
from unirl.utils.dtypes import parse_torch_dtype

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("unirl.tools.precompute_minimax_h3")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute MiniMax-H3 text embeddings (Qwen3-VL layer 50).")
    parser.add_argument("--model-path", help="MiniMax-H3 checkpoint or text_encoder repo.")
    parser.add_argument(
        "--data-path",
        required=True,
        nargs="+",
        help="Prompt files (.jsonl, .json, or .txt); pass every split the run reads, eval included.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for index.json and safetensors shards.")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--check-coverage", action="store_true")
    parser.add_argument("--check-parity", action="store_true")
    parser.add_argument("--parity-samples", type=int, default=5)
    parser.add_argument("--parity-atol", type=float, default=1e-3)
    parser.add_argument("--parity-rtol", type=float, default=1e-3)
    args = parser.parse_args(argv)
    if not args.check_coverage and not args.model_path:
        parser.error("--model-path is required unless --check-coverage")
    return args


def _load_prompts(data_paths: List[str], prompt_key: str) -> List[str]:
    prompts = (
        item["prompt"]
        for data_path in data_paths
        for item in TextPromptDataset(file_path=data_path, prompt_key=prompt_key).samples
    )
    return list(dict.fromkeys(prompts))


def _load_text_encoder(model_path: str, device: torch.device, dtype: torch.dtype):
    from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

    subfolder = "text_encoder" if os.path.isdir(os.path.join(model_path, "text_encoder")) else None
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, subfolder="tokenizer" if os.path.isdir(os.path.join(model_path, "tokenizer")) else None
    )
    processor = AutoProcessor.from_pretrained(
        model_path, subfolder="processor" if os.path.isdir(os.path.join(model_path, "processor")) else None
    )
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(model_path, subfolder=subfolder, torch_dtype=dtype)
    return encoder.to(device).eval().requires_grad_(False), tokenizer, processor


def run_precomputation(args: argparse.Namespace) -> None:
    prompts = _load_prompts(args.data_path, args.prompt_key)
    logger.info("Loaded %d prompts from %s", len(prompts), ", ".join(args.data_path))

    if args.check_coverage:
        OfflineTextEmbedStore.from_dir(args.output_dir).verify_coverage_or_raise(
            prompts, context=", ".join(args.data_path)
        )
        logger.info("Coverage OK: %d prompts in %s", len(prompts), args.output_dir)
        return

    writer = OfflineTextEmbedWriter(
        output_dir=args.output_dir,
        model_checkpoint=args.model_path,
        dtype=args.dtype,
        shard_size=args.shard_size,
        resume=args.resume,
        force_overwrite=args.force_overwrite,
    )
    needed = [prompt for prompt in prompts if not writer.contains(prompt)] if args.resume else list(prompts)
    logger.info("Need embeddings for %d/%d prompts", len(needed), len(prompts))
    if not needed and not args.check_parity:
        writer.close()
        return

    device = torch.device(args.device)
    dtype = parse_torch_dtype(args.dtype, field_name="precompute.dtype")
    text_encoder, tokenizer, processor = _load_text_encoder(args.model_path, device, dtype)

    def extract(prompt: str) -> torch.Tensor:
        hidden = encode_minimax_h3_prompt(
            text_encoder=text_encoder, tokenizer=tokenizer, processor=processor, prompt=prompt, device=device
        )
        return hidden.squeeze(0).detach().to("cpu", dtype=dtype).contiguous()

    started = time.perf_counter()
    for idx, prompt in enumerate(needed):
        writer.add(prompt, extract(prompt))
        if (idx + 1) % 100 == 0 or (idx + 1) == len(needed):
            elapsed = time.perf_counter() - started
            rate = (idx + 1) / elapsed if elapsed else 0.0
            logger.info("Processed %d/%d prompts (%.1f prompts/s)", idx + 1, len(needed), rate)
    writer.close()

    if not args.check_parity:
        return
    store = OfflineTextEmbedStore.from_dir(args.output_dir)
    peak = 0.0
    failed = False
    for prompt in prompts[: args.parity_samples]:
        cached = store.get(prompt).embeds.squeeze(0).float()
        live = extract(prompt).float()
        diff = (cached - live).abs().max().item()
        peak = max(peak, diff)
        if not torch.allclose(cached, live, atol=args.parity_atol, rtol=args.parity_rtol):
            logger.error("Parity FAILED for %r (max diff %.6e)", prompt, diff)
            failed = True
    if failed:
        raise RuntimeError(f"Numerical parity check failed; peak abs diff {peak:.6e}")
    logger.info("Parity PASSED; peak abs diff %.6e", peak)


def main() -> None:
    run_precomputation(parse_args())


if __name__ == "__main__":
    main()
