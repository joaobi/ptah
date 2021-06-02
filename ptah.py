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

class Ptah:
    def __init__(self, out_folder, fonts_folder, width,heigh,BCK_COLOR):
        # Instance Variable
        self.template_out_folder = out_folder
        self.fonts_folder = fonts_folder

        self.fontfiles = self._load_unicode_fonts()
        self.sign_codes,self.sign_hex_values = self._load_signs()
        self.width, self.height = width,heigh
        self.BCK_COLOR = BCK_COLOR

    def generate_template_images(self):
        font_sizes = [20,50,75,150,200,250,300,350,400,450]
        print('-------------------------------------------')
        print(' Generating template images')
        print('-------------------------------------------')

        t = trange(len(self.sign_codes), desc='Bar desc', leave=True)
        for i in t:
            # t.set_description("Bar desc (file %i)" % i, refresh=True, refresh=True)
            sign = self.sign_codes[i]

            t.set_description("Processing sign %s..." % sign, refresh=True)
            
            # print('Processing [%s]...'%(sign))
            for font_filename in self.fontfiles:
                for size_sign in font_sizes:
                    operations = ['','bc','crop']
                    for op in operations:                        
                        filename = os.path.join(self.template_out_folder,sign+"_"+font_filename.split('.')[0]+"_"+str(size_sign)+"_"+op+".jpg")
                        if not os.path.exists(filename): # only generate image if not already there
                            self._generate_image(sign,self.sign_hex_values[i],font_filename,size_sign,op)            
 

    def generate_train_val_images(self,output_folder):
        print('-------------------------------------------')
        print(' Generating train and val datasets')
        print('-------------------------------------------')
        t = trange(len(self.sign_codes), desc='Bar desc', leave=True)
        for i in t:
            sign = self.sign_codes[i]

            t.set_description("Processing sign %s..." % sign, refresh=True)

            # print('Processing [%s]...'%(sign))       
            # Need to Try Exception here and create dir if needed
            files = os.listdir(os.path.join(self.template_out_folder+'/'))
            result = [f for f in files if re.search(sign+'_', f)] 

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
                return
            (x, y, w,h)  = bb
            im = im.crop((x, y, x + w, y + h))
            # im_cropped.save("fonts-data/"+letter+"_"+font_name+"_"+str(font_size)+"_crop.jpg")            
        else:
            draw.text(((self.width-w)/2,(self.height-h)/2), glyph_text, font = font, fill="black")
        
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

        print("Loading %s gardiner signs..."%(len(codes)))

        return codes,hex_values

    def _save_image(self,im,sign,font_filename,font_size,operation,w,h):
         # Write to disk if image fits in canvas
        if (h<=self.height and w<=self.width):
            # Write to disk
            filename = sign+"_"+font_filename.split('.')[0]+"_"+str(font_size)+"_"+operation+".jpg"
            # print(filename)

            # Create output folder if it does not already exist
            try:
               _f = os.listdir(self.template_out_folder)
            except OSError as e:
                print(" Folder '%s'  does not exist, creating folder before saving..."%(self.template_out_folder))
                os.mkdir(self.template_out_folder)
                pass

            # Save template image to disk
            im.save(os.path.join(self.template_out_folder,filename))       

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


