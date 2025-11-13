// PARAMETERS
SOURCE = "//path/to/source/dir/"
OUTPUT_DIR = "//path/to/target/dir/";
Dataset = "Unique_Dataset_Identifier";

//Initial and End time point of the time-lapse for which masks will be generated
Tini = 0; 
Tend = 63;

// ============

t = Tini;
open(SOURCE+Dataset+"_T"+t+"_C=2.tiff");
run("Plot Z-axis Profile");

Dialog.create("Input thresholds");
Dialog.addMessage("Zero slice number:");
Dialog.addNumber("Slice number:", 10);
Dialog.show();
ZERO_SLICE = Dialog.getNumber();

run("Close All");

for (t=Tini; t<=Tend; t++) {

//	open(SOURCE+Dataset+"_T"+t+"_C=2.tiff");
	run("Bio-Formats Importer", "open="+SOURCE+Dataset+"_T"+t+"_C=2.tiff");
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
