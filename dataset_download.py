"""Download selected HEST datasets from Hugging Face."""

import argparse
import os
from typing import List, Optional

import datasets
import pandas as pd


DATASET_CONFIGS = {
    "her2st": {
        "data_path": "her2st",
        "meta_filter": {
            "dataset_title": "Spatial deconvolution of",
        },
        "note": "Suggested test slide: SPA148 (B1) or SPA123 (G2)",
    },
    "mouse_brain": {
        "data_path": "mouse_brain",
        "meta_filter": {
            "dataset_title": "Spatial Multimodal Analysis",
            "disease_state": "Healthy",
        },
        "note": "Suggested test slide: NCBI667",
    },
    "PRAD": {
        "data_path": "PRAD",
        "ids_to_query": [f"MEND{index}" for index in range(139, 163)],
        "note": "IDs MEND139-MEND162; suggested test slide: MEND145",
    },
    "kidney": {
        "data_path": "kidney",
        "meta_filter": {
            "dataset_title": (
                "Spatial localization with Spatial Transcriptomics for an "
                "atlas of healthy and injured cell states"
            ),
        },
        "note": "Suggested test slide: NCBI697",
    },
    "ccRCC": {
        "data_path": "ccRCC",
        "ids_to_query": [f"INT{index}" for index in range(1, 25)],
        "note": "IDs INT1-INT24; suggested test slide: INT2",
    },
}


def download_dataset(
    dataset_name: str,
    data_path: str,
    ids_to_query: Optional[List[str]] = None,
    meta_df: Optional[pd.DataFrame] = None,
    meta_filter: Optional[dict] = None,
    hf_dataset: str = "MahmoodLab/hest",
    hf_token: Optional[str] = None,
):
    """Download a filtered HEST dataset with cache reuse."""

    os.makedirs(data_path, exist_ok=True)

    if ids_to_query is None and meta_filter and meta_df is not None:
        mask = pd.Series(True, index=meta_df.index)
        for key, value in meta_filter.items():
            if key == "dataset_title" and isinstance(value, str):
                mask &= meta_df[key].str.startswith(value, na=False)
            else:
                mask &= meta_df[key] == value
        ids_to_query = meta_df.loc[mask, "id"].tolist()

    if not ids_to_query:
        raise ValueError(
            "No slide IDs found. Provide IDs or a valid metadata filter."
        )

    patterns = [f"*{slide_id}[_.]**" for slide_id in ids_to_query]
    print(f"Downloading {dataset_name}: {len(ids_to_query)} slide IDs")
    print(f"Destination: {data_path}")

    return datasets.load_dataset(
        hf_dataset,
        cache_dir=data_path,
        patterns=patterns,
        trust_remote_code=True,
        download_mode="reuse_cache_if_exists",
        token=hf_token,
    )


def get_dataset_config(dataset_name, base_path="./hest1k_datasets"):
    """Return a copy of the selected dataset configuration."""

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Choose from {list(DATASET_CONFIGS)}"
        )
    config = DATASET_CONFIGS[dataset_name].copy()
    config["data_path"] = os.path.join(
        base_path, config["data_path"]
    )
    return config


def main(
    dataset_name="her2st",
    hf_token=None,
    meta_version="v1_2_1",
    base_path="./hest1k_datasets",
):
    """Download one configured HEST cohort."""

    metadata_url = (
        f"hf://datasets/MahmoodLab/hest/HEST_{meta_version}.csv"
    )
    metadata = pd.read_csv(metadata_url)
    config = get_dataset_config(dataset_name, base_path)
    print(config.get("note", ""))

    return download_dataset(
        dataset_name=dataset_name,
        data_path=config["data_path"],
        ids_to_query=config.get("ids_to_query"),
        meta_df=metadata if "meta_filter" in config else None,
        meta_filter=config.get("meta_filter"),
        hf_token=hf_token,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Download HEST data used by GeneFlow."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS),
        default="her2st",
        help="HEST cohort to download.",
    )
    parser.add_argument(
        "--base_path",
        default="./hest1k_datasets",
        help="Directory that will contain the selected cohort.",
    )
    parser.add_argument(
        "--meta_version",
        default="v1_2_1",
        help="HEST metadata version.",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    main(
        dataset_name=arguments.dataset,
        hf_token=os.getenv("HF_TOKEN"),
        meta_version=arguments.meta_version,
        base_path=arguments.base_path,
    )
