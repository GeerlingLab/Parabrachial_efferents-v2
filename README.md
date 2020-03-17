Tested using Python 3.6.8, numpy 1.16.4, scipy 1.2.1, pillow 6.0.0, keras 2.2.4, tensorflow 1.12.0
Tested using 20x EFI Images from Olympus Slide Scanner as outlined in paper

This program uses a two-step method to detect synaptic boutons on histological images
The first step, a proposal algorithm, quickly scans through the image and detects potential boutons by looking for 
groups of pixels that are darker than the surrounding tissue

The second step, a verification algorithm, scans each of these potential boutons and verifies them via a pretrained
convolutional neural network.

Command line arguments allow the user to select how the program will search for input images
--folder: Intended for a single histological section.  Required format of this folder is as follows
    Folder
        Image001.tif  <-- RGB or Greyscale microscopy image.  Should be 345 nm/pixel
        Image002.tif  <-- Image001, Image002,... will be stitched together horizontally
        Mask.jpg <-- Optional Mask with same resolution as Image.jpg.  Boutons will only be found in white areas
--directory: Intended for an entire brain with many histological sections.  Required format
    Directory
        Section_01
            Image.jpg
            Mask.jpg
        Section_02
            Image.jpg
            Mask.jpg
        ...
--order: Intended for multiple brains, each with many histological sections.  If the brains are organized as so
    Order.txt
    Brain_01
        Section_01
            Image.jpg
            Mask.jpg
        ...
    Brain_02
        Section_01
            Image.jpg
            Mask.jpg
        ...
    ...
    Then Order.txt should read as so:
    Brain_01
    Brain_02
    ...        

Command line arguments allow the user to select three different types of output.  These output files will be placed
in the folder where the Image.jpg file was located
--png: Program will output a png file with the same dimensions as the image file, with red 5x5 boutons placed on it
--svg: Program will output a svg file with bouton symbols placed on it
--csv: Program will output a csv file with the x,y positions of each bouton

