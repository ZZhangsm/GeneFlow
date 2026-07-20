"""Dataset assembly helpers for GeneFlow."""

import os

import anndata
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class CustomDataset(Dataset):
    """Store expression vectors, image conditions, and optional coordinates."""

    def __init__(self, x, y, coords=None):
        self.data = x
        self.labels = y
        self.coords = coords

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.coords is not None:
            return self.data[idx], self.labels[idx], self.coords[idx]
        return self.data[idx], self.labels[idx]


def prepare_dataloader(args, dataset, batch_size, shuffle=True):
    sampler = DistributedSampler(
        dataset, shuffle=shuffle, seed=args.global_seed
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
    )


def build_path(data_path, *parts):
    return os.path.join(data_path, *parts)


def load_image_embeddings(data_path, sample_name, is_augmented=False):
    """Load and concatenate UNI and CONCH embeddings for one slide."""

    if is_augmented:
        uni_path = build_path(
            data_path,
            "processed_data",
            "1spot_uni_ebd_aug",
            f"{sample_name}_uni_aug.pt",
        )
        conch_path = build_path(
            data_path,
            "processed_data",
            "1spot_conch_ebd_aug",
            f"{sample_name}_conch_aug.pt",
        )
        concat_axis = -1
    else:
        uni_path = build_path(
            data_path,
            "processed_data",
            "1spot_uni_ebd",
            f"{sample_name}_uni.pt",
        )
        conch_path = build_path(
            data_path,
            "processed_data",
            "1spot_conch_ebd",
            f"{sample_name}_conch.pt",
        )
        concat_axis = 1

    uni_embedding = torch.load(uni_path, map_location="cpu")
    conch_embedding = torch.load(conch_path, map_location="cpu")
    return torch.cat([uni_embedding, conch_embedding], dim=concat_axis)


def load_count_matrix(data_path, sample_name, selected_genes):
    """Load the selected gene count matrix for one slide."""

    adata_path = build_path(data_path, "st", f"{sample_name}.h5ad")
    adata = anndata.read_h5ad(adata_path)
    adata.var_names_make_unique()
    matrix = adata[:, selected_genes].X
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    return pd.DataFrame(
        matrix,
        columns=selected_genes,
        index=[f"{sample_name}_{index}" for index in range(adata.shape[0])],
    )


def load_string_list(path):
    values = np.genfromtxt(path, dtype=str)
    return [str(value) for value in np.atleast_1d(values).tolist()]


def assemble_dataset(input_args):
    """Load original and augmented embeddings for all training slides."""

    processed_dir = build_path(input_args.data_path, "processed_data")
    slide_list_path = build_path(
        processed_dir, input_args.folder_list_filename
    )
    slide_names = load_string_list(slide_list_path)

    held_out_slides = [
        name.strip()
        for name in input_args.slide_out.split(",")
        if name.strip()
    ]
    for slide_name in held_out_slides:
        slide_names.remove(slide_name)
        input_args.logger.info("%s is held out for testing.", slide_name)

    input_args.logger.info(
        "Remaining %d slides: %s", len(slide_names), slide_names
    )

    gene_list_path = build_path(
        processed_dir, input_args.gene_list_filename
    )
    selected_genes = load_string_list(gene_list_path)
    input_args.input_gene_size = len(selected_genes)
    input_args.logger.info(
        "Selected genes: %s (%d)",
        input_args.gene_list_filename,
        len(selected_genes),
    )

    count_matrices = []
    original_embeddings = []
    for sample_name in slide_names:
        count_matrix = load_count_matrix(
            input_args.data_path, sample_name, selected_genes
        )
        image_embedding = load_image_embeddings(
            input_args.data_path, sample_name, is_augmented=False
        )
        count_matrices.append(count_matrix)
        original_embeddings.append(image_embedding)
        input_args.logger.info(
            "%s loaded: counts %s, image embeddings %s",
            sample_name,
            count_matrix.shape,
            tuple(image_embedding.shape),
        )

    all_counts_original = pd.concat(
        count_matrices, axis=0, ignore_index=True
    ).to_numpy()
    all_embeddings_original = torch.cat(original_embeddings, dim=0)
    input_args.cond_size = all_embeddings_original.shape[1]

    augmented_embeddings = [
        load_image_embeddings(
            input_args.data_path, sample_name, is_augmented=True
        )
        for sample_name in slide_names
    ]
    all_embeddings_augmented = torch.cat(
        augmented_embeddings, dim=0
    )

    augmentation_ratio = input_args.num_aug_ratio
    available_augmentations = all_embeddings_augmented.shape[1]
    if augmentation_ratio > available_augmentations:
        raise ValueError(
            "num_aug_ratio cannot exceed the number of augmented patches "
            f"per spot ({available_augmentations})."
        )

    all_counts_augmented = np.repeat(
        all_counts_original, augmentation_ratio, axis=0
    )
    selected_embeddings_augmented = torch.empty(
        (
            all_counts_augmented.shape[0],
            all_embeddings_augmented.shape[2],
        )
    )
    for spot_index in range(all_embeddings_augmented.shape[0]):
        selected_indices = np.random.choice(
            available_augmentations,
            augmentation_ratio,
            replace=False,
        )
        start = spot_index * augmentation_ratio
        end = start + augmentation_ratio
        selected_embeddings_augmented[start:end] = (
            all_embeddings_augmented[spot_index, selected_indices]
        )

    all_embeddings = torch.cat(
        [all_embeddings_original, selected_embeddings_augmented], dim=0
    )
    all_counts = np.concatenate(
        [all_counts_original, all_counts_augmented], axis=0
    )

    count_frame = pd.DataFrame(all_counts, columns=selected_genes)
    invalid_rows = count_frame.isnull().all(axis=1) | (
        count_frame.sum(axis=1) == 0
    )
    valid_indices = np.flatnonzero(~invalid_rows.to_numpy())
    count_frame = count_frame.iloc[valid_indices]
    all_embeddings = all_embeddings[valid_indices]

    normalized_counts = np.log2(
        count_frame.loc[:, selected_genes] + 1
    )
    all_embeddings.requires_grad_(False)
    input_args.logger.info(
        "Final dataset: counts %s, image embeddings %s",
        normalized_counts.shape,
        tuple(all_embeddings.shape),
    )

    dataset = CustomDataset(
        torch.from_numpy(normalized_counts.to_numpy()).float(),
        all_embeddings.float(),
    )
    return dataset, input_args
