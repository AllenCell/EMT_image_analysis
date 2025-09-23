import os
import yaml
import argparse
import logging
import copy

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCALE_PATCH_SHAPES = {
    1: [16, 128, 128],
    2: [16, 256, 256],
    3: [16, 512, 512]
}

def count_parts(data_dir: str, csv_base_name: str) -> int:
    return len([
        f for f in os.listdir(data_dir)
        if f.startswith(csv_base_name) and f.endswith('.csv') and '_p' in f
    ])

def generate_yaml_files(
    scale: int,
    batch_size: int,
    save_dir: str,
    start_index: int = 1,
    template_yaml: str = None,
    data_dir: str = None,
    csv_base_name: str = None
) -> None:
    if not template_yaml or not os.path.isfile(template_yaml):
        raise FileNotFoundError(f"Template YAML '{template_yaml}' not found.")
    if not data_dir or not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")
    if scale not in SCALE_PATCH_SHAPES:
        raise ValueError(f"Invalid scale {scale}. Must be one of {list(SCALE_PATCH_SHAPES.keys())}.")
    if not csv_base_name:
        raise ValueError("csv_base_name must be provided.")

    output_yaml_prefix = f"{csv_base_name}_scale{scale}_p"
    patch_shape = SCALE_PATCH_SHAPES[scale]

    with open(template_yaml, 'r') as f:
        template_data = yaml.safe_load(f)

    num_parts = count_parts(data_dir, csv_base_name)
    if num_parts < start_index:
        raise ValueError(f"No CSV parts found starting from index {start_index} in {data_dir}")

    os.makedirs(save_dir, exist_ok=True)

    for i in range(start_index, num_parts + 1):
        csv_name = f"{csv_base_name}_p{i}.csv"
        yaml_path = os.path.join(save_dir, f"{output_yaml_prefix}{i}.yaml")
        yaml_data = copy.deepcopy(template_data)
        yaml_data['data']['csv_path'] = f"${{paths.data_dir}}/{csv_name}"
        yaml_data['model']['save_dir'] = f"scale{scale}"
        yaml_data['data']['batch_size'] = batch_size
        yaml_data['data'].setdefault('_aux', {})['patch_shape'] = patch_shape
        with open(yaml_path, 'w') as out_f:
            out_f.write("# @package _global_\n\n")
            yaml.dump(yaml_data, out_f, sort_keys=False, default_flow_style=False)
        logging.info(f"Created: {yaml_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation YAMLs for cyto-dl.")
    parser.add_argument("--scale", type=int, choices=[1, 2, 3], required=True, help="Scale level: 1, 2, or 3 (required)")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for each YAML config (required)")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save generated YAMLs (default: same as data_dir)")
    parser.add_argument("--start_index", type=int, default=1, help="Start index for CSV parts (default: 1)")
    parser.add_argument("--template_yaml", type=str, required=True, help="Path to the base template YAML (required)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing chunked CSV files (required)")
    parser.add_argument("--csv_base_name", type=str, required=True, help="Base name for chunked CSV files (required)")
    args = parser.parse_args()
    # If save_dir is not provided, use data_dir
    save_dir = args.save_dir if args.save_dir else args.data_dir
    generate_yaml_files(
        scale=args.scale,
        batch_size=args.batch_size,
        save_dir=save_dir,
        start_index=args.start_index,
        template_yaml=args.template_yaml,
        data_dir=args.data_dir,
        csv_base_name=args.csv_base_name,
    )

if __name__ == "__main__":
    main()

