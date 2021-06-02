import ptah as pt

if __name__ == '__main__': 
    gen = pt.Ptah("template_images/","fonts",320,320,(255, 255, 255))

    # Step 1: generate the template images based on the unicode fonts
    gen.generate_template_images()

    # Step 2: generate the training and valiation datasets
    gen.generate_train_val_images('data')