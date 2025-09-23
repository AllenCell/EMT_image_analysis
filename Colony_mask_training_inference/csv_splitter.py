import pandas as pd
import os
import argparse
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def split_csv_chunks(chunk_size: int = 5, filename: str = None) -> None:
    """
    Split the CSV file into chunks.
    Args:
        chunk_size (int): Number of rows per chunk.
        filename (str): Path to the CSV file to split.
    """
    if not filename or not os.path.isfile(filename):
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
        description="Split CSV file into smaller chunks."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=3,
        help="Number of rows per chunk (default: 3)"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to the CSV file to split"
    )
    return parser.parse_args()

def main() -> None:
    """
    Main function to execute the script.
    """
    args = parse_args()
    split_csv_chunks(args.chunk_size, args.csv_file)

if __name__ == "__main__":
    main()
