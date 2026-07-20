"""Shared helpers for GeneFlow training."""

from collections import OrderedDict
import json
import logging
import os

import torch
import torch.distributed as dist


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Move the EMA model parameters toward the current model."""

    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(
            param.data, alpha=1 - decay
        )


def requires_grad(model, flag=True):
    """Set the requires_grad flag for every model parameter."""

    for parameter in model.parameters():
        parameter.requires_grad = flag


def cleanup():
    """End distributed training if a process group is active."""

    if dist.is_initialized():
        dist.destroy_process_group()


def create_logger(logging_dir):
    """Create a rank-aware logger for stdout and an optional log file."""

    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = logging.getLogger("GeneFlow")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if rank == 0:
        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if logging_dir is not None:
            file_handler = logging.FileHandler(
                os.path.join(logging_dir, "log.txt")
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())

    return logger


def save_args_config(
    args, experiment_dir, logger=None, filename="config.json"
):
    """Save JSON-serializable training arguments."""

    config = {}
    for key, value in vars(args).items():
        try:
            json.dumps(value)
            config[key] = value
        except TypeError:
            config[key] = str(value)

    config_path = os.path.join(experiment_dir, filename)
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    if logger is not None:
        logger.info("Saved arguments to %s", config_path)
    return config
