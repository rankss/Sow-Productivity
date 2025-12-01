import argparse
import pandas as pd
import numpy as np
from utils import create_directory

def synthesize_dataset(dataset_type, n_sows=100, max_parities=7, n_farms=5, random_state=None):
    """
    Generates a synthetic dataset mimicking the structure of 'cdpq' or 'hypor' raw data.

    Args:
        dataset_type (str): The type of dataset to synthesize, either 'cdpq' or 'hypor'.
        n_sows (int, optional): The number of unique sows to generate. Defaults to 100.
        max_parities (int, optional): The maximum number of consecutive parities for any sow.
            Defaults to 7.
        n_farms (int, optional): The number of farms to simulate for the 'hypor' dataset.
            Defaults to 5.
        random_state (int, optional): Seed for the random number generator for reproducibility.
            Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the synthetic data.

    Raises:
        ValueError: If an unsupported dataset_type is provided.
    """
    rng = np.random.default_rng(random_state)
    records = []

    for sow_id in range(1, n_sows + 1):
        num_parities = rng.integers(1, max_parities + 1)
        for parity in range(1, num_parities + 1):
            record = {
                "Sow ID": f"Sow_{sow_id}",
                "Parity": parity,
                "Gestation Length": rng.normal(115, 2),
                "Lactation Length": rng.normal(21, 3),
                "Stillborn": rng.poisson(1.2),
                "Mummies": rng.poisson(0.5),
                "Piglets Weaned": rng.integers(9, 14),
                "Liveborn": rng.integers(10, 18)
            }
            records.append(record)

    df = pd.DataFrame(records)

    # Add dataset-specific columns and formatting
    if dataset_type == 'cdpq':
        df["Farm"] = 1
        # Simulate body weight and backfat measurements
        df["Breeding Weight"] = rng.normal(230, 25, size=len(df))
        df["Farrowing Weight"] = df["Breeding Weight"] + rng.normal(30, 10, size=len(df))
        df["Weaning Weight"] = df["Farrowing Weight"] - rng.normal(20, 8, size=len(df))
        df["Breeding Backfat"] = rng.normal(17, 3, size=len(df))
        df["Farrowing Backfat"] = df["Breeding Backfat"] + rng.normal(2, 1, size=len(df))
        df["Weaning Backfat"] = df["Farrowing Backfat"] - rng.normal(1.5, 1, size=len(df))

    elif dataset_type == 'hypor':
        # Simulate multiple farms
        sow_farm_map = {f"Sow_{sow_id}": f"{rng.integers(1, n_farms + 1)}" for sow_id in range(1, n_sows + 1)}
        df["Farm"] = df["Sow ID"].map(sow_farm_map)

    else:
        raise ValueError(f"Unsupported dataset_type: '{dataset_type}'. Choose 'cdpq' or 'hypor'.")

    # Round numeric columns to reasonable precision
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].round(2)
        else:
            df[col] = df[col].astype(int)

    create_directory("raw_data")
    df.to_excel(f"raw_data/{dataset_type}_raw_dataset.xlsx", index=False)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset mimicking 'cdpq' or 'hypor' structure.")
    parser.add_argument("dataset_type", type=str, choices=['cdpq', 'hypor'], help="The type of dataset to synthesize.")
    parser.add_argument("--n_sows", type=int, default=100, help="The number of unique sows to generate.")
    parser.add_argument("--max_parities", type=int, default=7, help="The maximum number of consecutive parities for any sow.")
    parser.add_argument("--n_farms", type=int, default=5, help="The number of farms to simulate for the 'hypor' dataset.")
    parser.add_argument("--random_state", type=int, default=None, help="Seed for the random number generator.")

    args = parser.parse_args()

    synthesize_dataset(
        dataset_type=args.dataset_type,
        n_sows=args.n_sows,
        max_parities=args.max_parities,
        n_farms=args.n_farms,
        random_state=args.random_state
    )