import os
import yaml
import argparse
import logging
import copy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Constants
SCALE_PATCH_SHAPES = {
    1: [16, 128, 128],
    2: [16, 256, 256],
    3: [16, 512, 512]
}

DEFAULT_TEMPLATE_YAML = 'eval_scale1_7521_p1.yaml'
DEFAULT_DATA_DIR = '../../../data/'

def count_parts(data_dir: str, csv_base_name: str) -> int:
    """
    Count how many chunked CSV files exist in the given directory.

    Args:
        data_dir (str): Path to the directory containing CSV files.
        csv_base_name (str): Base name of the CSV files.

    Returns:
        int: Number of matching CSV files.
    """
    return len([
        f for f in os.listdir(data_dir)
        if f.startswith(csv_base_name) and f.endswith('.csv') and '_p' in f
    ])

def generate_yaml_files(
    movie_num: str,
    scale: int,
    batch_size: int,
    save_dir: str,
    start_index: int = 1,
    template_yaml: str = DEFAULT_TEMPLATE_YAML,
    data_dir: str = DEFAULT_DATA_DIR
) -> None:
    """
    Generate YAML files for evaluation based on a template.

    Args:
        movie_num (str): Movie number (e.g., '7523').
        scale (int): Scale level (1, 2, or 3).
        batch_size (int): Batch size for YAMLs.
        save_dir (str): Directory to save generated YAMLs.
        start_index (int): Start index for CSV parts.
        template_yaml (str): Path to the base template YAML.
        data_dir (str): Directory containing the input CSVs.

    Raises:
        FileNotFoundError: If template YAML or data directory doesn't exist.
        ValueError: If scale is invalid or no CSV parts found.
    """
    if not os.path.isfile(template_yaml):
        raise FileNotFoundError(f"Template YAML '{template_yaml}' not found.")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    if scale not in SCALE_PATCH_SHAPES:
        raise ValueError(f"Invalid scale {scale}. Must be one of {list(SCALE_PATCH_SHAPES.keys())}.")

    csv_base_name = f"{movie_num}_all_moviepaths_qc"
    output_yaml_prefix = f"eval_scale{scale}_{movie_num}_p"
    patch_shape = SCALE_PATCH_SHAPES[scale]

    # Load template YAML
    with open(template_yaml, 'r') as f:
        template_data = yaml.safe_load(f)

    num_parts = count_parts(data_dir, csv_base_name)
    if num_parts < start_index:
        raise ValueError(f"No CSV parts found starting from index {start_index} in {data_dir}")

    os.makedirs(save_dir, exist_ok=True)

    for i in range(start_index, num_parts + 1):
        csv_name = f"{csv_base_name}_p{i}.csv"
        yaml_path = os.path.join(save_dir, f"{output_yaml_prefix}{i}.yaml")

        # Deep copy to avoid mutation across iterations
        yaml_data = copy.deepcopy(template_data)

        # Update fields
        yaml_data['data']['csv_path'] = f"${{paths.data_dir}}/{csv_name}"
        yaml_data['model']['save_dir'] = f"/allen/aics/emt/all_cells_mask/{movie_num}_scale{scale}"
        yaml_data['data']['batch_size'] = batch_size
        yaml_data['data'].setdefault('_aux', {})['patch_shape'] = patch_shape

        # Write YAML with hydra-compatible header
        with open(yaml_path, 'w') as out_f:
            out_f.write("# @package _global_\n\n")
            yaml.dump(yaml_data, out_f, sort_keys=False, default_flow_style=False)

        logging.info(f"Created: {yaml_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation YAMLs for cyto-dl.")
    parser.add_argument("--movie_num", type=str, required=True, help="Movie number (e.g., 7523)")
    parser.add_argument("--scale", type=int, choices=[1, 2, 3], required=True, help="Scale level (1, 2, or 3)")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for each YAML config")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save generated YAMLs")
    parser.add_argument("--start_index", type=int, default=1, help="Start index for CSV parts (default: 1)")
    parser.add_argument("--template_yaml", type=str, default=DEFAULT_TEMPLATE_YAML, help="Path to the base template YAML")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing chunked CSV files")

    args = parser.parse_args()

    generate_yaml_files(
        movie_num=args.movie_num,
        scale=args.scale,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        start_index=args.start_index,
        template_yaml=args.template_yaml,
        data_dir=args.data_dir,
    )

if __name__ == "__main__":
    main()

