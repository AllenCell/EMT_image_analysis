# Instructions to run the fiji macro for colony mask groundtruth generation
  # Run specific variables
    # Images should be split time point-wise and channel-wise 
    # In SOURCE, provide the source directory containing CytoGFP tagged images
    # In OUTPUT_DIR provide the target direcotry for the generated masks
    # Dataset corresponds to the common identifier associated with all images under consideration 
    # `T_ini` and `T_end` are the start and end timepoints for the mask generation
  # How to run?
    # Open Fiji
    # Drag and drop the *.jim file on it and press Run or go to `Plugins --> Macros --> Run --> select *.jim`
    # A pop-up window will appear with plotted Z-axis profile
    # Select the slice number as the start point of the maximum slope region (point from where Z-profile has the highest slope)
    # Click Ok and the macro will run (ignore the images that will appear and close on Fiji)
  
    
