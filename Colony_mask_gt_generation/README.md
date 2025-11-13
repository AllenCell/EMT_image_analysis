# Instructions to Run the Fiji Macro for ColonyMask Groundtruth Generation

This method has been validated on TIFF images. Follow the steps below to ensure proper execution.

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
