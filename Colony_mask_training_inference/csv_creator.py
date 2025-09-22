import os
import csv
import glob
import argparse
import sys
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

def create_csv_for_movie(base_root: str, output_dir: str = '.') -> bool:
    """
    Generates a CSV listing all .ome.zarr files for the specified movie.

    Args:
        base_root (str): Root directory containing movie folders.
        output_dir (str): Directory where the output CSV will be saved.

    Returns:
        bool: True if CSV was created successfully, False otherwise.
    """
    zarr_files = glob.glob(os.path.join(base_root, '*.ome.zarr'))

    if not zarr_files:
        logging.warning(f"No .ome.zarr files found in {base_root}")
        return False

    output_csv = os.path.join(output_dir, "all_moviepaths_qc.csv")

    try:
        with open(output_csv, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['count', 'movie_path', 'bf_channel'])  # bf_channel is set to 0 for all rows
            for idx, path in enumerate(sorted(zarr_files)):
                norm_path = os.path.normpath(path)
                writer.writerow([idx, norm_path, 0])
    except IOError as e:
        logging.error(f"Error writing CSV file {output_csv}: {e}")
        return False

    logging.info(f"CSV file created: {output_csv}")
    return True

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CSV of .ome.zarr files."
    )
    parser.add_argument(
        '--base_path',
        type=str,
        help="Base directory containing movie files (default: %(default)s)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help="Directory to save the output CSV (default: current directory)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    success = create_csv_for_movie(args.base_path, args.output_dir)
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()

