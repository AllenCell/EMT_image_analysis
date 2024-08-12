/*
 * Dialogue box for user input
 */

Dialog.create("Colony mask groundtruth generation using CytoGFP images");
	Dialog.addString("Source folder:", "//allen/aics/microscopy/Data/RnD_Sandbox/3500006256_20240223_Deliverable_ZSD0/ch2/", 128);
	Dialog.addString("Output folder:", "//allen/aics/assay-dev/users/Suraj/EMT_Work/CytoGFP/GT_Generation/otsu_thresholding_test/", 128);
	Dialog.addString("Dataset identifier:", "3500006256_20240223_20X_Timelapse_scene_29", 128);
	Dialog.addNumber("T ini:", 0);
	Dialog.addNumber("T end:", 63);
Dialog.show();

SOURCE_DIR = Dialog.getString();
OUTPUT_DIR = Dialog.getString();
DATASET = Dialog.getString();
T_ini = Dialog.getNumber();
T_end = Dialog.getNumber();

/*
 * Finding frame with fastest intensity change
 */

t = T_ini;
open(SOURCE_DIR+DATASET+"_T"+t+"_C=2.tiff");
run("Plot Z-axis Profile");

Dialog.create("Input thresholds");
Dialog.addMessage("Zero slice number:");
Dialog.addNumber("Slice number:", 10);
Dialog.show();
ZERO_SLICE = Dialog.getNumber();

run("Close All");

/*
 * Processing all images with same identifier
 */

for (t=T_ini; t<=T_end; t++) {

	run("Bio-Formats Importer", "open="+SOURCE_DIR+DATASET+"_T"+t+"_C=2.tiff");
	run("Duplicate...", "duplicate");
	setOption("BlackBackground", true);
	setAutoThreshold("Otsu dark no-reset stack");
	run("Convert to Mask", "background=Dark black");
	
	run("Median 3D...", "x=3 y=3 z=1");
	
	run("Select All");
	setForegroundColor(255, 255, 255);
	setBackgroundColor(0, 0, 0);
	for (z=1; z<=ZERO_SLICE; z++) {
		setSlice(z);
		run("Clear", "slice");
	}
	
	name = getInfo("window.title");
	name = substring(name, 0, lengthOf(name)-12);
	saveAs("Tiff", OUTPUT_DIR+name);
	run("Close All");

}