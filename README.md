# EMT Image Analysis Code

This repository contains the code for reproducing the image analysis pipeline described in our manuscript [1]. The pipeline uses publicly available data hosted on AWS to perform segmentation and 3D mesh generation. The outputs of this analysis are further utilized by the companion repository, **[EMT_data_analysis](https://github.com/AllenCell/EMT_data_analysis)**, to generate the plots and visualizations presented in the manuscript.

[1] - [A human induced pluripotent stem (hiPS) cell model for the holistic study of epithelial to mesenchymal transitions (EMTs)](https://www.biorxiv.org/content/10.1101/2024.08.16.608353v1)

---

## Workflow Components

1. **CytoGFP Groundtruth Segmentation**  
   - This module generates ground truth segmentations for CytoGFP-tagged images.  
   - It is used to create accurate masks for training and validating segmentation models.  
   - [Detailed Instructions](./Colony_mask_gt_generation/README.md)

2. **All Cells Mask Model Training and Inference**  
   - This component trains a deep learning model to predict all cell masks from label-free images.  
   - It also supports inference on new datasets to generate all cell masks.  
   - [Detailed Instructions](./Colony_mask_training_inference/README.md)

3. **H2B and EOMES Nuclear Segmentations**  
   - Performs instance segmentation of H2B and EOMES nuclear markers.  
   - This step is critical for identifying individual nuclei.  
   - [Detailed Instructions](./H2B_and_EOMES_instance_segmentation/README.md)

4. **CollagenIV Segmentation**  
   - Segments CollagenIV structures from the provided images.  
   - This step is essential for analyzing extracellular matrix organization.  
   - [Detailed Instructions](./CollagenIV_segmentation/README.md)

5. **CollagenIV Segmentation Mesh Generation**  
   - Converts CollagenIV segmentations into 3D meshes for downstream analysis.  
   - These meshes are used to study the structural properties of the extracellular matrix.  
   - [Detailed Instructions](./CollagenIV_mesh_generation/README.md)

---

## Contact

If you have questions about this code, please reach out to us at **cells@alleninstitute.org**.

---

## Licensing

All code in this repository is provided under the **Allen Institute Software License**.
