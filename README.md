# Ptah - Generate Egyptian hieroglyphs dataset for OCR

This projects is meant to provide tools to create training and validation datasets for bulding OCR models for Egyptian hieroglyphs.  

Output images are generated by writing the hieroglyplic letters signs to images using a set of unicode fonts that support hieroglyphs and by performing a set of permutaions on these. 

## Requirements
### Unicode Fonts
Download and unzip the unicode fonts and place them in the "fonts" directory.

Here is the list of fonts I am using:
Font Name | Link
--- | ---
Abydos |https://dn-works.com/ufas/
Aegyptus  |https://dn-works.com/wp-content/uploads/2020/UFAS-Fonts/Aegyptus.zip
JSesh  |http://files.qenherkhopeshef.org/jsesh/JSeshFont.ttf
NewGardiner  |https://mjn.host.cs.st-andrews.ac.uk/egyptian/fonts/newgardiner.html
NotoSansEgyptianHieroglyph | https://github.com/googlefonts/noto-fonts/blob/main/hinted/ttf/NotoSansEgyptianHieroglyphs/NotoSansEgyptianHieroglyphs-Regular.ttf
Segoe UI Historic  |Available on Windows 10 devices
Sinuhe |https://github.com/somiyagawa/SINUHE-the-Hierotyper/blob/master/fonts/webfonts/SINUHE.ttf
Aaron fonts| https://github.com/HieroglyphsEverywhere/Fonts/tree/master/Experimental



More details at: https://github.com/HieroglyphsEverywhere/Fonts/blob/master/HieroglyphicFontList.md.

  
### Python dependencies
tba

## File Structure

hierogenerator.py - calls the data generator  
ptah.py - class that does all ther work  
ptahlibs.py - file with the Gardiner codes (sign and unicode)  
/fonts - location where to place the uncompressed fonts  
/template-images - output directory for the template signs. Will be created automatically.  
/data - output directory for the training and validation dataset. Will be created automatically.  
/demo - (future) location for demo apps


