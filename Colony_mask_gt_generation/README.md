# Instructions to Run the Fiji Macro for ColonyMask Groundtruth Generation

## 🧬 Dataset Format and Usage

This method has been **validated on 3D TIFF image stacks**, where each TIFF file represents a **single z-stack** corresponding to one **timepoint**.  
Unlike the **multi-timepoint z-stack sequences** provided in the [OME-ZARR format](https://open.quiltdata.com/b/allencell/tree/aics/emt_timelapse_dataset/), each TIFF used here contains only a single volumetric acquisition.

To ensure compatibility with this Fiji macro, the original OME-ZARR files must first be **split by timepoint** and saved following the naming convention below:

Where:  
- **`Unique_Dataset_Identifier`** — an optional, user-defined tag to distinguish datasets or experimental conditions  
- **`TXX`** — the timepoint index (e.g., `T01`, `T15`, etc.)  
- **`C=2`** — denotes the imaging channel of interest (as used in this workflow)  

---

## ⚙️ Execution Instructions

1. **Split the OME-ZARR dataset** into separate TIFF stacks, one per timepoint.  
2. **Rename each file** according to the format above.  
3. **Place all TIFFs** in the designated input directory recognized by the Fiji macro.  
4. **Run the macro** following the standard Fiji execution steps for this repository.  

This configuration enables consistent preprocessing of volumetric image data for downstream quantitative analysis of 3D cellular structures.

---

## 1. Prerequisites
- **Image Requirements**:
  - Images should be split **time point-wise** and **channel-wise**.
- **Input Variables**:
  - `SOURCE`: Directory containing CytoGFP-tagged images.
  - `OUTPUT_DIR`: Target directory for the generated masks.
  - `Dataset`: Common identifier associated with all images under consideration.
  - `Tini` and `Tend`: Start and end timepoints for mask generation.

---

## 2. How to Run the Macro?

1. **Open Fiji**:
   - Ensure Fiji is installed on your system. If not, download it from [Fiji Downloads](https://imagej.net/software/fiji/).

2. **Load the Macro**:
   - Drag and drop the `*.jim` file onto the Fiji interface and press **Run**.
   - Alternatively, navigate to:
     - `Plugins` → `Macros` → `Run` → Select the `*.jim` file.

3. **Z-Axis Profile Selection**:
   - A pop-up window will appear with the plotted Z-axis profile.
   - Identify the slice number corresponding to the **start point of the maximum slope region**:
     - This is the point where the Z-profile has the **highest slope**.
   - Enter the slice number and click **OK**.

4. **Macro Execution**:
   - The macro will now run.
   - Ignore any images that appear and close automatically in Fiji.

---

## 3. Notes
- Ensure the `SOURCE` and `OUTPUT_DIR` directories are correctly set before running the macro.
- The macro is designed to handle CytoGFP-tagged images. For other image types, modifications may be required.
- For troubleshooting, refer to the Fiji documentation or the macro's error logs.

---

## 4. Example Command
```bash
# Example of setting variables in the macro
SOURCE=/path/to/source/images
OUTPUT_DIR=/path/to/output/masks
Dataset=example_dataset
Tini=1
Tend=10
```
