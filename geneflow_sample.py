"""Sampling entry point for GeneFlow."""

import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from GeneFlow.data import CustomDataset, load_image_embeddings
from GeneFlow.flow.interpolant import Interpolant
from GeneFlow.models import GeneFlow_models


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def find_model(model_path, device="cpu"):
    """Load EMA weights when available, otherwise model weights."""

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {model_path}"
        )
    checkpoint = torch.load(model_path, map_location=device)
    if "ema" in checkpoint:
        return checkpoint["ema"]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


@torch.no_grad()
def sample_expression(args, interpolant, model, loader):
    predictions = []
    for batch in tqdm(loader, desc="Sampling"):
        _, image_features = [
            tensor.to(args.device) for tensor in batch
        ]
        expression = interpolant.sample(
            model,
            image_features,
            input_gene_size=args.input_gene_size,
            n_sample_steps=args.n_sample_steps,
        )
        predictions.append(expression.detach().cpu())
    return torch.cat(predictions, dim=0)


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    model = GeneFlow_models[args.model](
        input_size=args.input_gene_size,
        depth=args.DiT_num_blocks,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        label_size=args.cond_size,
        token_dim=args.token_dim,
    )
    model.load_state_dict(
        find_model(args.ckpt, device=args.device)
    )
    model.to(args.device)
    model.eval()

    interpolant = Interpolant(
        args.prior_sampler,
        total_count=torch.tensor([args.zinb_total_count]),
        logits=torch.tensor([args.zinb_logits]),
        zi_logits=args.zinb_zi_logits,
        normalize=args.prior_sampler != "gaussian",
    )
    loader = DataLoader(
        args.dataset,
        batch_size=args.sampling_batch_size,
        shuffle=False,
    )

    start_time = time.time()
    samples = sample_expression(
        args, interpolant, model, loader
    )
    total_time = time.time() - start_time

    output_dir = Path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = Path(args.ckpt).stem
    output_path = output_dir / (
        f"generated_{checkpoint_stem}"
        f"_cond{args.sample_num_per_cond}"
        f"_n{args.n_sample_steps}.pt"
    )
    torch.save(samples, output_path)
    print(f"Saved {len(samples)} samples to {output_path}")
    print(f"Total sampling time: {total_time:.2f} seconds")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate spatial expression with GeneFlow."
    )
    parser.add_argument(
        "--model",
        choices=list(GeneFlow_models),
        default="GeneFlow",
    )
    parser.add_argument("--DiT_num_blocks", type=int, default=12)
    parser.add_argument("--hidden_size", type=int, default=24)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--token_dim", type=int, default=100)

    parser.add_argument("--n_sample_steps", type=int, default=5)
    parser.add_argument(
        "--prior_sampler",
        choices=["gaussian", "zero", "zinb"],
        default="zinb",
    )
    parser.add_argument("--zinb_logits", type=float, default=0.1)
    parser.add_argument("--zinb_total_count", type=float, default=1)
    parser.add_argument("--zinb_zi_logits", type=float, default=0.0)

    parser.add_argument("--slide_out", default="MEND145")
    parser.add_argument(
        "--gene_list_filename",
        default="selected_gene_list_1k.txt",
    )
    parser.add_argument(
        "--sample_num_per_cond", type=int, default=20
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampling_batch_size", type=int, default=1024
    )

    parser.add_argument(
        "--save_path", default="./results/samples/"
    )
    parser.add_argument(
        "--ckpt", required=True, help="Checkpoint to load."
    )
    parser.add_argument(
        "--data_path",
        default="./hest1k_datasets/PRAD/",
    )
    parser.add_argument("--device", default="cuda")
    return parser


def prepare_arguments(args):
    image_embeddings = load_image_embeddings(
        args.data_path, args.slide_out, is_augmented=False
    )
    args.raw_cond = image_embeddings
    args.cond_size = image_embeddings.shape[1]
    args.cond = image_embeddings.repeat_interleave(
        args.sample_num_per_cond, dim=0
    )

    gene_list_path = os.path.join(
        args.data_path,
        "processed_data",
        args.gene_list_filename,
    )
    selected_genes = np.atleast_1d(
        np.genfromtxt(gene_list_path, dtype=str)
    )
    args.input_gene_size = len(selected_genes)
    args.dataset = CustomDataset(args.cond, args.cond)
    return args


if __name__ == "__main__":
    arguments = prepare_arguments(
        build_parser().parse_args()
    )
    main(arguments)
