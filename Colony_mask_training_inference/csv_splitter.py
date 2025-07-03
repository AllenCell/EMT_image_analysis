import pandas as pd
import os
import argparse
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def split_csv_chunks(movie_number: str, chunk_size: int = 5, base_dir: str = '.') -> None:
    """
    Split the CSV file for the given movie number into chunks.

    Args:
        movie_number (str): The movie number prefix for the CSV file.
        chunk_size (int): Number of rows per chunk.
        base_dir (str): Directory where the CSV file is located.
    """
    filename = os.path.join(base_dir, f"{movie_number}_all_moviepaths_qc.csv")

    if not os.path.isfile(filename):
        logging.error(f"File '{filename}' does not exist.")
        raise FileNotFoundError(f"File '{filename}' does not exist.")

    df = pd.read_csv(filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        part_num = (i // chunk_size) + 1
        out_filename = f"{base_name}_p{part_num}.csv"
        chunk.to_csv(out_filename, index=False)
        logging.info(f"Saved: {out_filename}")

def parse_args(): 
    parser = argparse.ArgumentParser(
        description="Split movie CSV file into smaller chunks by movie number."
    )
    parser.add_argument(
        "movie_number",
        type=str,
        help="Movie number prefix, e.g. 7287"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=3,
        help="Number of rows per chunk (default: 3)"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default='.',
        help="Directory where CSV files are stored (default: current directory)"
    )
    return parser.parse_args()

def main() -> None:
    """
    Main function to execute the script.
    """
    args = parse_args()
    split_csv_chunks(args.movie_number, args.chunk_size, args.base_dir)

if __name__ == "__main__":
    main()
