import os
import argparse
import numpy as np
from PIL import Image
import pandas as pd

'''
Generates config files- 1 for each movie for snakemake - Pass in the FMS manifest(from Antoine) and the gene name of the gene you want to process
'''

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv_dir", type= str, default="/allen/aics/assay-dev/computational/data/EMT_deliverable_processing/Collagen_segmentation_segmentations/EMT_deliverable_1_dataset_1_with_aligned_path_info.csv")
parser.add_argument("--input_log_dir", type= str, default="/allen/aics/assay-dev/computational/data/EMT_deliverable_processing/Collagen_segmentation_segmentations/Snakemake_logs/EOMES_TBR2/input_logs")
parser.add_argument("--gene_name", type= str, default="EOMES|TBR2")

if __name__ == "__main__":
    args = parser.parse_args()
    input_csv = pd.read_csv(args.input_csv_dir)
    subset = input_csv[(input_csv['Gene']==args.gene_name)  & (input_csv["EMT Condition"] == "3D MG EMT 1:60 MG")]
    ground_truth_list = list(subset["file_id"].values)
    print(ground_truth_list)
    # Generate .txt file for each thing in ground_truth_list

    for i in range(len(ground_truth_list)):
        with open(os.path.join(args.input_log_dir, f"fmsid_{ground_truth_list[i]}_input.txt"), 'w') as f:
            f.write(f"{ground_truth_list[i]}") 

            