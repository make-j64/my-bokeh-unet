#!/usr/bin/python3

import argparse
import os
import random
import logging
import cv2
import math
from scipy import misc

import tensorflow as tf
import numpy as np
from PIL import Image

from unet import UNet, Discriminator
from scripts.image_manips import resize

model_name = "matting"

logging.basicConfig(level=logging.INFO)

def image_fill(img, size, value):

    border = [math.ceil((size[0] - img.shape[0])/2),
              math.floor((size[0] - img.shape[0])/2),
              math.ceil((size[1] - img.shape[1])/2),
              math.floor((size[1] - img.shape[1])/2)]
    return cv2.copyMakeBorder(img,border[0],border[1],border[2],border[3],cv2.BORDER_CONSTANT,value=value)

def load_image(image_file):
    size = [960/2, 720/2]

    image = cv2.imread(image_file, cv2.IMREAD_COLOR)

    ratio = np.amin(np.divide(size, image.shape[0:2]))
    image_size = np.floor(np.multiply(image.shape[0:2], ratio)).astype(int)
    image = cv2.resize(image, (image_size[1], image_size[0]))
    image = image_fill(image,size,[0,0,0,0])
    image = image.astype(float)
    return image

def generate_trimap(object_file):
    size = [960/2, 720/2]

    foreground = cv2.imread(object_file, cv2.IMREAD_UNCHANGED)
    if foreground is None:
        return False
    print(foreground.shape)
    alpha = cv2.split(foreground)[3]

    ratio = np.amin(np.divide(size, alpha.shape[0:2]))
    forground_size = np.floor(np.multiply(alpha.shape[0:2], ratio)).astype(int)
    alpha = cv2.resize(alpha, (forground_size[1], forground_size[0]))
    alpha = image_fill(alpha,size,[0,0,0,0])

    alpha = alpha.astype(float)
    cv2.normalize(alpha, alpha, 0.0, 1.0, cv2.NORM_MINMAX)

    _, inner_map = cv2.threshold(alpha, 0.9, 255, cv2.THRESH_BINARY)
    _, outer_map = cv2.threshold(alpha, 0.1, 255, cv2.THRESH_BINARY)

    inner_map = cv2.erode(inner_map, np.ones((5,5),np.uint8), iterations = 3)
    outer_map = cv2.dilate(outer_map, np.ones((5,5),np.uint8), iterations = 3)

    return inner_map + (outer_map - inner_map) /2

# Parse Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Evalutate image")
    parser.add_argument("input", type=str,
        help="Path to a file containing input image")
    parser.add_argument("object", type=str,
        help="Path to a file containing trimap image")
    parser.add_argument("output", type=str,
        help="Path to the output file")
    parser.add_argument('--checkpoint', type=int, default=None,
        help='Saved session checkpoint, -1 for latest.')
    parser.add_argument('--logdir', default="log/" + model_name,
        help='Directory where logs should be written.')
    return parser.parse_args()


def apply_trimap(images, output, alpha):
    masked_output = []
    for channel in range(4):
        masked_output.append(output[:,:,:,channel])
        masked_output[channel] = tf.where(alpha < 0.25, images[:,:,:,channel], masked_output[channel])
        masked_output[channel] = tf.where(alpha > 0.75, images[:,:,:,channel], masked_output[channel])
        masked_output[channel] = masked_output[channel]
    masked_output = tf.stack(masked_output, 3)
    return masked_output

def main(args):
    input_images = tf.placeholder(tf.float32, shape=[1, 480, 360, 4])

    with tf.variable_scope("Gen"):
        gen = UNet(4,3)
        output = tf.sigmoid(gen(input_images))

    global_step = tf.get_variable('global_step', initializer=0, trainable=False)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()
    if args.checkpoint is not None and os.path.exists(os.path.join(args.logdir, 'checkpoint')):
        if args.checkpoint == -1:#latest checkpoint
            saver.restore(sess, tf.train.latest_checkpoint(args.logdir))
        else:#Specified checkpoint
            saver.restore(sess, os.path.join(args.logdir, model_name+".ckpt-"+str(args.checkpoint)))
        logging.info('Model restored to step ' + str(global_step.eval(sess)))


    images, targets = [], []

    I = misc.imread('/Users/hiepth/Desktop/Data_Bokeh/ebb_dataset/train/original/000000010100.jpg')
    I_depth = misc.imread('/Users/hiepth/Desktop/Data_Bokeh/ebb_dataset/train/original_depth/000000010100.png')
    #print('namemmmm: '+name+ " num "+str(i))

    I = cv2.resize(I, (360, 480));
    I_depth = cv2.resize(I_depth, (360, 480));
    # Stacking the image together with its depth map
    I_temp = np.zeros((I.shape[0], I.shape[1], 4))
    I_temp[:, :, 0:3] = I
    I_temp[:, :, 3] = I_depth
    I = I_temp


    # Extracting random patch of width PATCH_WIDTH

    I = np.float32(I) / 255.0
    result = sess.run(output, feed_dict={
            input_images: np.asarray([I]),
            })

    print(result.shape)
    image = Image.fromarray((result[0,:,:,:]*255).astype(np.uint8))
    image.save('output_'+str(args.checkpoint)+'.jpg')


if __name__ == '__main__':
    args = parse_args()
    main(args)
