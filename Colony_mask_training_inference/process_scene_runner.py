import pandas as pd
import os
import subprocess
import re
import argparse
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional, Tuple, Set
import time
import logging

# Constants
POLL_INTERVAL = 0.1  # seconds
DEFAULT_PROCESS_SCRIPT = 'process_scene.py'

def setup_logging() -> None:
    """Set up logging for the main process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] [Main] %(message)s',
        handlers=[
            logging.FileHandler('progress.log', mode='w'),  # Overwrite log file
            logging.StreamHandler()
        ]
    )

def extract_scene_name(filepath: str) -> Optional[str]:
    """Extract the scene name from the file path."""
    start = filepath.find('(')
    end = filepath.find(')')

    if start != -1 and end != -1:
        scene_name = filepath[start + 1:end]
        return scene_name
    else:
        logging.warning(f"Could not extract scene name from: {filepath}")
        return None

def run_process(cmd: list[str], scene: str, output_queue: Queue) -> None:
    """Run a subprocess and store its output."""
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        output_queue.put((scene, process.returncode, stdout, stderr))
    except Exception as e:
        output_queue.put((scene, -1, "", f"Failed to start process: {str(e)}"))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process scenes in parallel')
    parser.add_argument('--max-processes', type=int, default=4,
                       help='Maximum number of concurrent processes. Default: 4')
    parser.add_argument('--csv-path', type=str, required=True,
                       help='Path to the CSV file containing movie paths (/path/to/EMT_image_analysis/Colony_mask_training_inference/data/all_moviepaths_qc.csv)')
    parser.add_argument('--barcode', type=str, required=True,
                       help='Barcode number for path construction')
    parser.add_argument('--base-path', type=str, required=True,
                       help=f'Base path for ACM outputs (output directory with barcode subdirectories as /all_cells_mask/<barcode>/infer_<barcode>_multiscale_v2/)')
    args = parser.parse_args()
    
    setup_logging()
    logging.info("Starting main process")

    # Use the parsed arguments
    csv_path = args.csv_path
    barcode = args.barcode
    base_path = args.base_path.rstrip('/')  # Remove trailing slash if present
    
    # Construct paths using the base path and barcode parameters
    target_base = f'{base_path}/{barcode}/infer_{barcode}_multiscale_v2/'
    path_512 = f'{base_path}/{barcode}/scale3/seg/'
    path_128 = f'{base_path}/{barcode}/scale1/seg/'
    path_scale2 = f'{base_path}/{barcode}/scale2/seg/'
    
    logging.info(f"Using base path: {base_path}")
    logging.info(f"Using barcode: {barcode}")
    logging.info(f"Target base directory: {target_base}")
    
    # Validate input file exists
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return
    
    # Read and validate CSV file
    try:
        df = pd.read_csv(csv_path)
        if 'movie_path' not in df.columns:
            logging.error("CSV file must contain 'movie_path' column")
            return
        logging.info(f"Successfully loaded CSV file: {csv_path}")
        logging.info(f"CSV contains {len(df)} rows")
    except Exception as e:
        logging.error(f"Failed to read CSV file: {e}")
        return

    scenes = set()
    skipped_scenes = 0
    for filepath in df['movie_path']:
        scene = extract_scene_name(filepath)
        if scene:
            scenes.add(scene)
            scene_dir = os.path.join(target_base, scene)
            try:
                os.makedirs(scene_dir, exist_ok=True)
            except OSError as e:
                logging.error(f"Failed to create directory {scene_dir}: {e}")
                return
        else:
            skipped_scenes += 1
    
    if skipped_scenes > 0:
        logging.warning(f"Skipped {skipped_scenes} rows due to invalid scene names")
    
    if not scenes:
        logging.error("No valid scenes found in CSV file")
        return
    
    total_scenes = len(scenes)
    logging.info(f"Found {total_scenes} scenes to process")

    # Set the maximum number of concurrent processes
    # Default to number of scenes (no limit) if not specified by user
    MAX_PROCESSES = args.max_processes if args.max_processes is not None else len(scenes)
    logging.info(f"Maximum concurrent processes: {MAX_PROCESSES}")
    
    active_processes = []
    output_queue = Queue()
    
    # Start one process per scene
    for i, scene in enumerate(scenes, 1):
        cmd = [
            'python', DEFAULT_PROCESS_SCRIPT,
            '--scene', scene,
            '--target-dir', target_base,
            '--path-512', path_512,
            '--path-128', path_128,
            '--path-scale2', path_scale2
        ]
        
        # Start process in a separate thread to handle output
        logging.info(f"Starting process for scene {scene} ({i}/{total_scenes})")
        process_thread = Thread(target=run_process, args=(cmd, scene, output_queue))
        process_thread.start()
        active_processes.append(process_thread)
        
        # If we've reached the max number of processes, wait for some to finish
        while len(active_processes) >= MAX_PROCESSES:
            active_processes = [p for p in active_processes if p.is_alive()]
            time.sleep(POLL_INTERVAL)
    
    # Wait for all processes to complete
    for process_thread in active_processes:
        process_thread.join()
    
    # Collect and print results
    completed_scenes = 0
    while not output_queue.empty():
        scene, returncode, stdout, stderr = output_queue.get()
        completed_scenes += 1
        logging.info(f"Scene {scene} completed ({completed_scenes}/{total_scenes}) with return code {returncode}")
        if stdout:
            logging.info(f"Scene {scene} stdout:\n{stdout}")
        if stderr:
            logging.info(f"Scene {scene} stderr:\n{stderr}")

    logging.info("All scenes processed")

if __name__ == '__main__':
    main()
