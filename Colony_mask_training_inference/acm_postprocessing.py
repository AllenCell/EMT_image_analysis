import pandas as pd
from pathlib import Path
import logging
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('generate_movie_mapping.log')
    ]
)
logger = logging.getLogger(__name__)


def get_data(item: str) -> tuple[int, int]:
    """
    Extract position index and well label from a string formatted as 'P1-B2'.
    """
    try:
        prefix, suffix = item.split('-')
        p_num = int(prefix[1:])
        b_num = int(suffix[1:])
        return p_num, b_num
    except Exception as e:
        logger.error(f"Failed to parse scene identifier '{item}': {e}")
        raise


def extract_movie_name(movie_path: str) -> str:
    """
    Extract movie name (e.g., P1-B2) from file name in path with format: '.../(P1-B2)...'
    """
    try:
        filename = Path(movie_path).stem
        start = filename.find('(')
        end = filename.find(')')
        if start == -1 or end == -1:
            raise ValueError("Parentheses not found in file name.")
        return filename[start + 1:end]
    except Exception as e:
        logger.error(f"Error extracting movie name from {movie_path}: {e}")
        raise


def load_input_csv(barcode: str) -> pd.DataFrame:
    csv_path = f'{barcode}_qc.csv'
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} movies from {csv_path}")
    return df


def create_acm_mapping(df: pd.DataFrame, barcode: str) -> pd.DataFrame:
    base_acm_path = f'/allen/aics/emt/all_cells_mask/{barcode}/infer_{barcode}_multiscale'
    df['ACM_path'] = df['movie_path'].apply(
        lambda path: f"{base_acm_path}/{extract_movie_name(path)}"
    )
    return df[['movie_path', 'ACM_path']].rename(columns={'movie_path': 'Raw_movie_path'})


def add_plate_barcode_column(df: pd.DataFrame) -> None:
    try:
        df['Plate Barcode'] = df['Raw_movie_path'].apply(
            lambda x: Path(x).stem.split('_')[0]
        )
        logger.info("Added 'Plate Barcode' column.")
    except Exception as e:
        logger.error(f"Error adding 'Plate Barcode' column: {e}")


def add_metadata_columns(df: pd.DataFrame) -> None:
    for i, path in enumerate(df['ACM_path']):
        try:
            scene_name = Path(path).stem
            pid, wlabel = get_data(scene_name)
            df.at[i, 'Position Index'] = pid
            df.at[i, 'Well Label'] = wlabel
            logger.info(f"Processed scene: {scene_name} - Position Index: {pid}, Well Label: {wlabel}")
        except Exception as e:
            logger.warning(f"Skipping row due to error: {e}")


def save_output(df: pd.DataFrame, barcode: str, suffix: str = 'final') -> None:
    out_path = f'{barcode}_qc_run_{suffix}.csv'
    df.to_csv(out_path, index=False)
    logger.info(f"Saved output to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate movie-to-ACM mapping CSV with metadata.")
    parser.add_argument('--barcode', required=True, help="Plate barcode (e.g., 7450)")
    args = parser.parse_args()

    # Step 1: Load and map ACM paths
    df = load_input_csv(args.barcode)
    mapped_df = create_acm_mapping(df, args.barcode)
    save_output(mapped_df, args.barcode, suffix='run')

    # Step 2: Load the saved mapping and add further metadata
    metadata_df = pd.read_csv(f'{args.barcode}_qc_run.csv')
    add_plate_barcode_column(metadata_df)
    add_metadata_columns(metadata_df)
    save_output(metadata_df, args.barcode, suffix='final') 

if __name__ == '__main__':
    main()

