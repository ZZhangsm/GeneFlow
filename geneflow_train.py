"""Distributed training entry point for GeneFlow."""

import argparse
from copy import deepcopy
from glob import glob
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from GeneFlow.data import assemble_dataset, prepare_dataloader
from GeneFlow.flow.interpolant import Interpolant
from GeneFlow.models import GeneFlow_models
from GeneFlow.train_helper import (
    create_logger,
    requires_grad,
    save_args_config,
    update_ema,
)

try:
    import swanlab
except ImportError:
    swanlab = None


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        rank: int,
        gpu_id: int,
        model_args: argparse.Namespace,
    ):
        self.rank = rank
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.args = model_args

        self.model = model
        self.ema = deepcopy(model).to(gpu_id)
        requires_grad(self.ema, False)

        self.interpolant = Interpolant(
            self.args.prior_sampler,
            total_count=torch.tensor([self.args.zinb_total_count]),
            logits=torch.tensor([self.args.zinb_logits]),
            zi_logits=self.args.zinb_zi_logits,
            normalize=self.args.prior_sampler != "gaussian",
        )
        self.model = DDP(
            self.model.to(gpu_id), device_ids=[self.gpu_id]
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.lr, weight_decay=0
        )
        update_ema(self.ema, self.model.module, decay=0)

        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        self.args.logger.info(
            "Rank %d initialized trainer with %.2f M parameters",
            rank,
            parameter_count / 1e6,
        )

        self.train_steps = 0
        self.log_steps = 0
        self.running_loss = 0
        self.use_swanlab = self.args.use_swanlab

    def run_batch(self, expression, image_embedding):
        loss = self.interpolant.flow_matching_loss(
            self.model, expression, image_embedding
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.args.clip_norm
        )
        self.optimizer.step()
        update_ema(self.ema, self.model.module)

        self.train_steps += 1
        if self.use_swanlab and self.rank == 0:
            swanlab.log(
                {
                    "train_loss": loss.item(),
                    "train_steps": self.train_steps,
                }
            )

        self.log_steps += 1
        self.running_loss += loss.item()
        if self.log_steps % 100 == 0:
            torch.cuda.synchronize()
            average_loss = torch.tensor(
                self.running_loss / self.log_steps,
                device=expression.device,
            )
            dist.all_reduce(average_loss, op=dist.ReduceOp.SUM)
            average_loss = (
                average_loss.item() / dist.get_world_size()
            )
            self.args.logger.info(
                "Step=%07d | Training loss: %.5f",
                self.train_steps,
                average_loss,
            )
            if self.use_swanlab and self.rank == 0:
                swanlab.log({"average_loss": average_loss})
            self.running_loss = 0
            self.log_steps = 0

        if self.train_steps % self.args.ckpt_every == 0:
            if self.rank == 0:
                self.save_checkpoint()
            dist.barrier()

    def run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[0])
        self.args.logger.info(
            "GPU %s | Epoch %d | Batch size %d | Steps %d",
            self.gpu_id,
            epoch,
            batch_size,
            len(self.train_data),
        )
        self.train_data.sampler.set_epoch(epoch)

        for expression, image_embedding in self.train_data:
            if self.train_steps >= self.args.total_steps:
                break
            self.run_batch(
                expression.to(self.gpu_id),
                image_embedding.to(self.gpu_id),
            )

        if self.use_swanlab and self.rank == 0:
            swanlab.log({"epoch": epoch + 1})

    def save_checkpoint(self):
        checkpoint = {
            "model": self.model.module.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.optimizer.state_dict(),
        }
        checkpoint_path = os.path.join(
            self.args.checkpoint_dir,
            f"{self.train_steps:07d}.pt",
        )
        torch.save(checkpoint, checkpoint_path)
        self.args.logger.info(
            "Saved checkpoint to %s", checkpoint_path
        )

    def train(self, max_steps):
        self.model.train()
        self.ema.eval()
        self.args.total_steps = int(max_steps)
        epoch = 0
        while self.train_steps < self.args.total_steps:
            self.run_epoch(epoch)
            epoch += 1

        if self.use_swanlab and self.rank == 0:
            swanlab.finish()


def load_train_objects(args):
    train_set, args = assemble_dataset(args)
    model = GeneFlow_models[args.model](
        input_size=args.input_gene_size,
        depth=args.DiT_num_blocks,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        label_size=args.cond_size,
        token_dim=args.token_dim,
    )
    args.logger.info(
        "Dataset contains %d samples from %s",
        len(train_set),
        args.data_path,
    )
    return train_set, model, args


def main(world_size, available_gpus, input_args):
    dist.init_process_group(backend="nccl", world_size=world_size)
    rank = dist.get_rank()
    device = available_gpus[rank]
    seed = input_args.global_seed * dist.get_world_size() + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    if rank == 0:
        os.makedirs(input_args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{input_args.results_dir}/*"))
        input_args.experiment_dir = os.path.join(
            input_args.results_dir, f"{experiment_index:03d}"
        )
        input_args.checkpoint_dir = os.path.join(
            input_args.experiment_dir, "checkpoints"
        )
        os.makedirs(input_args.checkpoint_dir, exist_ok=True)
        os.makedirs(
            os.path.join(input_args.experiment_dir, "samples"),
            exist_ok=True,
        )
        input_args.logger = create_logger(input_args.experiment_dir)
    else:
        input_args.logger = create_logger(None)

    input_args.logger.info(
        "Rank %d | Device %s | Seed %d", rank, device, seed
    )

    if rank == 0:
        config = save_args_config(
            input_args,
            input_args.experiment_dir,
            logger=input_args.logger,
        )
    else:
        config = None

    if input_args.use_swanlab and rank == 0:
        if swanlab is None:
            raise ImportError(
                "Install swanlab or disable --use_swanlab."
            )
        if not input_args.swanlab_note:
            raise ValueError(
                "--swanlab_note is required with --use_swanlab."
            )
        swanlab.init(
            project=(
                f"{input_args.swanlab_project}_"
                f"{input_args.expr_name}"
            ),
            name=(
                f"{os.path.basename(input_args.experiment_dir)}_"
                f"{input_args.swanlab_note}"
            ),
            config=config,
        )

    dataset, model, args = load_train_objects(input_args)
    train_data = prepare_dataloader(
        args,
        dataset,
        int(args.global_batch_size // dist.get_world_size()),
    )
    trainer = Trainer(
        model, train_data, rank, int(device.split(":")[-1]), args
    )
    trainer.train(args.total_steps)
    dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train GeneFlow with PyTorch DDP."
    )

    parser.add_argument("--expr_name", default="her2st")
    parser.add_argument(
        "--data_path",
        default="./hest1k_datasets/her2st/",
        help="Dataset root containing st/ and processed_data/.",
    )
    parser.add_argument(
        "--results_dir",
        default="./results/her2st/runs/",
        help="Directory used for experiment outputs.",
    )
    parser.add_argument("--slide_out", default="SPA148")
    parser.add_argument(
        "--folder_list_filename", default="all_slide_lst.txt"
    )
    parser.add_argument(
        "--gene_list_filename", default="selected_gene_list_1k.txt"
    )
    parser.add_argument("--num_aug_ratio", type=int, default=7)

    parser.add_argument(
        "--model",
        choices=list(GeneFlow_models),
        default="GeneFlow",
    )
    parser.add_argument("--DiT_num_blocks", type=int, default=12)
    parser.add_argument("--hidden_size", type=int, default=24)
    parser.add_argument("--token_dim", type=int, default=100)
    parser.add_argument("--num_heads", type=int, default=6)

    parser.add_argument(
        "--prior_sampler",
        choices=["gaussian", "zero", "zinb"],
        default="zinb",
    )
    parser.add_argument("--zinb_logits", type=float, default=0.1)
    parser.add_argument("--zinb_total_count", type=float, default=1)
    parser.add_argument("--zinb_zi_logits", type=float, default=0.0)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--total_steps", type=int, default=15000)
    parser.add_argument("--global_batch_size", type=int, default=512)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help=(
            "Number of DDP GPU processes. Match this value to "
            "torchrun --nproc_per_node."
        ),
    )
    parser.add_argument("--ckpt_every", type=int, default=5000)

    parser.add_argument("--use_swanlab", action="store_true")
    parser.add_argument(
        "--swanlab_project", default="GeneFlow"
    )
    parser.add_argument("--swanlab_note")

    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    process_count = arguments.num_workers
    gpu_devices = [
        f"cuda:{index}" for index in range(process_count)
    ]
    main(process_count, gpu_devices, arguments)
