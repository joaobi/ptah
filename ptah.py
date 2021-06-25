import os
import re
from PIL import Image,ImageFont,ImageDraw
import numpy as np
import cv2
import imutils
from pathlib import Path
from tqdm import tqdm, trange

import ptahlibs as sc
import sys

# For generating ligatures

# import matplotlib.pyplot as plt
# from matplotlib.path import Path as mpPath
# import matplotlib.patches as patches

from fontTools.ttLib import TTFont

from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from textwrap import dedent

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

class Ptah:
    def __init__(self, out_folder, fonts_folder, width,heigh,BCK_COLOR):
        # Instance Variable
        self.template_out_folder = out_folder
        self.fonts_folder = fonts_folder

        self.fontfiles = self._load_unicode_fonts()
        self.sign_codes,self.sign_hex_values,self.aegyptus_lig_values,self.egyptianhiero_values = self._load_signs()
        self.width, self.height = width,heigh
        self.BCK_COLOR = BCK_COLOR

    def generate_template_images(self):
        font_sizes = [20,50,75,150,200,250,300,350,400,450]
        print('-------------------------------------------')
        print(' Generating template images')
        print('-------------------------------------------')

        t = trange(len(self.sign_codes), desc='Bar desc', leave=True)
        for i in t:
            sign = self.sign_codes[i]

            t.set_description("Processing sign %s..." % sign, refresh=True)
            
            # print('Processing [%s]...'%(sign))
            for font_filename in self.fontfiles:
                # generate the SVG base image for this file
                if sign.startswith('ZZ'):
                    if font_filename in ['Aegyptus.otf','AegyptusBold.otf']:
                        code = self.aegyptus_lig_values[i]
                        self._gen_ligature_svg(sign,code,font_filename)
                    elif font_filename in ['EgyptianHiero4.03.ttf']:
                        code = self.egyptianhiero_values[i]
                        self._gen_ligature_svg(sign,code,font_filename)

                for size_sign in font_sizes:
                    operations = ['','bc','crop']
                    # operations = ['']
                    for op in operations:                        
                        filename = os.path.join(self.template_out_folder,sign+"_"+font_filename.split('.')[0]+"_"+str(size_sign)+"_"+op+".jpg")
                        if not os.path.exists(filename): # only generate image if not already there
                            if sign.startswith('ZZ'):
                                if font_filename in ['Aegyptus.otf','AegyptusBold.otf','EgyptianHiero4.03.ttf'] and op == '': # only generate ligatures once
                                    self._generate_ligature(sign,font_filename,size_sign,op)
                            else:
                                # print(filename+ ' does not exist! sign:'+sign+ ' op: '+op+' size: '+str(size_sign))  
                                self._generate_image(sign,self.sign_hex_values[i],font_filename,size_sign,op)
                        # else:
                        #     print(filename+ ' exists!')            
 

    def _gen_ligature_svg(self,sign,code,font_filename):
        char = code

        if font_filename in ['Aegyptus.otf','AegyptusBold.otf','EgyptianHiero4.03.ttf']:
            if font_filename in ['Aegyptus.otf','AegyptusBold.otf']:
                translate = ""
            else:
                translate = "translate(0 250)"

            font = TTFont(os.path.join(self.fonts_folder,font_filename), 0, allowVID=0,
                    ignoreDecompileErrors=True,
                    fontNumber=-1)

            glyph = font.getGlyphSet()[char]
            glyphSet = font.getGlyphSet()

            pen = DecomposingRecordingPen(glyphSet)
            glyph.draw(pen)

            bound_pen = BoundsPen(glyph)
            pen.replay(bound_pen)

            cmap = font.getBestCmap()

            svgpen = SVGPathPen(font)
            pen.replay(svgpen)
            # glyph.draw(svgpen)
            path = svgpen.getCommands()
            # print(path)

            ascender = font['OS/2'].sTypoAscender
            descender = font['OS/2'].sTypoDescender
            width = glyph.width
            height = ascender - descender

            content = dedent(f'''\
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {-ascender} {width} {height}">
                    <g transform="{translate} scale(1, -1)">
                        <path d="{path}"/>
                    </g>
                </svg>
            ''')
            filename = sign+"_"+font_filename.split('.')[0]+".svg"
            # print(filename)
            with open(os.path.join(self.template_out_folder,filename), 'w') as f:
                f.write(content)


    def _generate_ligature(self,sign,font_filename,size_sign,op):
        
        filename = sign+"_"+font_filename.split('.')[0]+".svg"

        drawing = svg2rlg(os.path.join(self.template_out_folder,filename))
        img = renderPM.drawToPIL(drawing)

        out_filename = sign+"_"+font_filename.split('.')[0]+"_"+str(size_sign)+".jpg"

        img.resize((size_sign,size_sign)).save(os.path.join(self.template_out_folder,out_filename))


    def generate_train_val_images(self,output_folder):
        print('-------------------------------------------')
        print(' Generating train and val datasets')
        print('-------------------------------------------')
        t = trange(len(self.sign_codes), desc='Bar desc', leave=True)
        for i in t:
            sign = self.sign_codes[i]

            t.set_description("Processing sign %s..." % sign, refresh=True)

            # print('Processing [%s]...'%(sign))       
            files = os.listdir(os.path.join(self.template_out_folder+'/'))
            result = [f for f in files if re.search('^'+sign+'_'+'.*jpg', f)] 

            dataset_size = len(result)
            n_val = round(dataset_size*0.3)
            n_train = dataset_size-n_val


            train_imgs = np.random.choice(result, size=n_train, replace=False)
            val_imgs = set(train_imgs) ^ set(result)

            self._create_images(output_folder,'train', train_imgs,sign)
            self._create_images(output_folder,'val', val_imgs,sign)



    def _create_images(self,output_folder,out_dir, images,sign):

        full_path = output_folder+'/'+out_dir+'/'+sign

        Path(full_path).mkdir(parents=True, exist_ok=True)
        for filename in images:
            if not os.path.exists(os.path.join(output_folder+'/'+out_dir+'/'+sign,filename+'_'+str(1)+".jpeg")): # only generate image if not already there
                image = Image.open(os.path.join(self.template_out_folder,filename))
                im_arr = np.array(image)

                img = image.convert("RGB")
                img.save(os.path.join(output_folder+'/'+out_dir+'/'+sign,filename+'_'+str(1)+".jpeg"))
                img.transpose(Image.FLIP_LEFT_RIGHT).save(
                    os.path.join(output_folder+'/'+out_dir+'/'+sign,filename+'_'+str(2)+"_t.jpeg"))

    def _generate_image(self,sign,hex_code,font_filename,font_size,operation = ''):

        # do not generate small images that were only meant for cropping
        if operation != 'crop' and font_size in [20,50,75]:
            return
        elif operation == 'crop' and font_size not in [20,50,75,200]:
            return

        im,w,h,glyph_text,draw,font = self._write_sign(sign,hex_code,font_filename,font_size)

        if operation == 'tc': # Top center
            draw.text(((self.width-w)/2,0-45), glyph_text, font = font, fill="black")
        elif operation == 'bc': # Bottom center
            draw.text(((self.width-w)/2,(self.height-h)), glyph_text, font = font, fill="black")
        elif operation == 'crop' and font_size in [20,50,75,200]: # Crop image
            draw.text(((self.width-w)/2,(self.height-h)/2), glyph_text, font = font, fill="black")

            # Extract bounding box and print copy with no canvas
            bb = self._get_bounding_box(im)
            
            if bb == []:
                # print('No BBs...')
                return
            (x, y, w,h)  = bb
            im = im.crop((x, y, x + w, y + h))
            # im_cropped.save("fonts-data/"+letter+"_"+font_name+"_"+str(font_size)+"_crop.jpg")            
        else:
            draw.text(((self.width-w)/2,(self.height-h)/2), glyph_text, font = font, fill="black")
        
        # print('Writing:'+sign+"_"+font_filename.split('.')[0]+"_"+str(font_size)+"_"+operation+".jpg")
        self._save_image(im,sign,font_filename,font_size,operation,w,h)

    def _load_unicode_fonts(self):
        try:
            _f = os.listdir(self.fonts_folder)
            if len(_f) == 0: # Empty directory
                raise Exception()
            elif [f for f in os.listdir(self.fonts_folder) if re.match(".*tf",f)] == []: # no font files in directory
                raise Exception()
        except OSError as e:
            print(" Folder '%s'  does not exist"%(self.fonts_folder))
            sys.exit()
        except Exception as e:
            # print(e)
            print("Folder '%s' does not contain any fonts"%(self.fonts_folder))
            sys.exit()


        font_files = os.listdir(self.fonts_folder)
        # result = [f for f in files if re.search(sign+'_', f)] 
        # print(font_files)

        # font_files = [f+sc.hiero_fonts[f]['ext'] for f in sc.hiero_fonts.keys() if sc.hiero_fonts[f]['ext']]
        # print(font_files)

        print("Loading %i fonts from folder '%s'..."%(len(font_files), self.fonts_folder))

        return font_files

    def _load_signs(self):
        # Load Gardiner Codes
        codes = list(sc.gardiner_dict.keys())
        hex_values = [sc.gardiner_dict[f]['hex'] for f in sc.gardiner_dict.keys() if sc.gardiner_dict[f]['hex']]
        egyptus_values = [ sc.gardiner_dict[f].get('Aegyptus') if sc.gardiner_dict[f].get('Aegyptus') else -1  for f in sc.gardiner_dict if sc.gardiner_dict[f]]
        egyptianhiero_values = [ sc.gardiner_dict[f].get('EgyptianHiero') if sc.gardiner_dict[f].get('EgyptianHiero') else -1  for f in sc.gardiner_dict if sc.gardiner_dict[f]]
        
        print("Loading %s gardiner signs..."%(len(codes)))

        return codes,hex_values,egyptus_values,egyptianhiero_values

    def _save_image(self,im,sign,font_filename,font_size,operation,w,h):
         # Write to disk if image fits in canvas
        if (h<=self.height and w<=self.width):
            # Write to disk
            filename = sign+"_"+font_filename.split('.')[0]+"_"+str(font_size)+"_"+operation+".jpg"

            # Create output folder if it does not already exist
            try:
               _f = os.listdir(self.template_out_folder)
            except OSError as e:
                print(" Folder '%s'  does not exist, creating folder before saving..."%(self.template_out_folder))
                os.mkdir(self.template_out_folder)
                pass

            # Save template image to disk
            im.save(os.path.join(self.template_out_folder,filename))
        # else:
        #     print('Does not fit in canvas')       

    def _write_sign(self,letter,code,font_name,font_size):
        im = Image.new(mode = "RGB", size = (self.width, self.height), color = self.BCK_COLOR )
        draw = ImageDraw.Draw(im)  

        font = ImageFont.truetype(os.path.join(self.fonts_folder,font_name), font_size)

        glyph_text = chr(code)

        w, h = draw.textsize(glyph_text, font=font)

        return im,w,h,glyph_text,draw,font

    def _get_bounding_box(self,im):
        '''
        Returns the bounding box coordinates of the source image.

                Parameters:
                        im (PIL Image): Image for which we want to beg the Bounding Boxes

                Returns:
                        x,y,w,h (int): list of x,y, width and heigh of bounding box
        '''
        bg = Image.new(mode = "RGB", size = (self.width, self.height), color = self.BCK_COLOR )
        bg = cv2.cvtColor(np.asarray(bg), cv2.COLOR_RGB2BGR)

        img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

        diff = cv2.absdiff(bg, img)

        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        

        for i in range(0, 3):
            dilated = cv2.dilate(gray.copy(), None, iterations= i+ 1)

        (T, thresh) = cv2.threshold(dilated, 3, 255, cv2.THRESH_BINARY)
        
        # find countours
        cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if cnts == []:
            return []
        
        # I am assuming the last one is the largest Area i.e. the full BB
        return cv2.boundingRect(cnts[len(cnts)-1])


