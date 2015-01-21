#!/usr/bin/env python

# Requires
#    - Pillow python module
#    - ImageMagick convert executable

import argparse
import subprocess
from glob import glob
from os import remove
from PIL import Image, ImageSequence

def read_image(path_to_image):
    """
    Read image from path.
    Returns image object.
    """
    img = Image.open(path_to_image)
    curr_format = img.format
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img.format = curr_format
    return img

def create_filter_layer(image, r, g, b):
    """
    Create a new image from the input.
    Adds a transparent layer with given color.
    """
    curr_format = image.format
    filter_layer = Image.new('RGBA', image.size, (r,g,b,128))
    new_img = Image.alpha_composite(image, filter_layer)
    new_img.format = curr_format
    return new_img

def create_partyhardized_gif(images, path_to_save):
    """
    Aggregates the images and creates the animated gif
    """
    count = 0
    tmp_prefix = "tmp_ph_"
    for image in images:
        image.save("%s%s.%s" % (tmp_prefix, count, image.format.lower()), image.format)
        count += 1
    cmd = "convert -delay 10 -loop 0 %s* %s" % (tmp_prefix, path_to_save)
    process = subprocess.Popen(cmd, shell = True)
    process.communicate()
    for to_remove in glob("%s*" % tmp_prefix):
	pass
        remove(to_remove)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Com'on! You know what it is!")
    parser.add_argument("source", type=str, help="Image to use for partyhardization")
    parser.add_argument("dest", type=str, help="Where to save partyhardized image")
    args = parser.parse_args()
    
    img = read_image(args.source)
    img_r = create_filter_layer(img, 255, 0, 0)
    img_g = create_filter_layer(img, 0, 255, 0)
    img_b = create_filter_layer(img, 0, 0, 255)
    create_partyhardized_gif([img, img_r, img_g, img_b], args.dest)

