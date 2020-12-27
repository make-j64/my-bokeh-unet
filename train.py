#!/usr/bin/python3

import argparse
import os
import random
import logging
from scipy import misc

import tensorflow as tf
import numpy as np
from PIL import Image

from unet import UNet, Discriminator
from scripts.image_manips import resize
import cv2

model_name = "matting"
TRAIN_IMAGES = 50

logging.basicConfig(level=logging.INFO)

# Parse Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Trains the unet")
    parser.add_argument("data", type=str,
        help="Path to a folder containing data to train")
    parser.add_argument("--lr", type=float, default=1.0,
        help="Learning rate used to optimize")
    parser.add_argument("--d_coeff", type=float, default=1.0,
        help="Discriminator loss coefficient")
    parser.add_argument("--gen_epoch", type=int, default=40,
        help="Number of training epochs")
    parser.add_argument("--disc_epoch", type=int, default=10,
        help="Number of training epochs")
    parser.add_argument("--adv_epoch", type=int, default=50,
        help="Number of training epochs")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=4,
        help="Size of the batches used in training")
    parser.add_argument('--checkpoint', type=int, default=None,
        help='Saved session checkpoint, -1 for latest.')
    parser.add_argument('--logdir', default="log/" + model_name,
        help='Directory where logs should be written.')
    return  parser.parse_args()


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
    input_path = os.path.join(args.data, "original")
    trimap_path = os.path.join(args.data, "original_depth")
    target_path = os.path.join(args.data, "bokeh")
    output_path = os.path.join(args.data, "output")
    print('input_path '+input_path)
    train_data_update_freq = args.batch_size
    test_data_update_freq = 50*args.batch_size
    sess_save_freq = 100*args.batch_size

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    ids = [[int(i) for i in os.path.splitext(filename)[0].split('_')] for filename in os.listdir(input_path)]
    print("numimage "+str(len(ids)))
    np.random.shuffle(ids)
    split_point = int(round(0.85*len(ids))) #using 70% as training and 30% as Validation
    train_ids = tf.get_variable('train_ids', initializer=ids[0:split_point], trainable=False)
    valid_ids = tf.get_variable('valid_ids', initializer=ids[split_point:len(ids)], trainable=False)

    global_step = tf.get_variable('global_step', initializer=0, trainable=False)

    g_iter = int(args.gen_epoch * int(train_ids.shape[0]))
    d_iter = int(args.disc_epoch * int(train_ids.shape[0]))
    a_iter = int(args.adv_epoch * int(train_ids.shape[0]))
    n_iter = g_iter+d_iter+a_iter
    print('n_iter ' + str(n_iter))

    input_images = tf.placeholder(tf.float32, shape=[TRAIN_IMAGES, 480, 360, 4])
    target_images = tf.placeholder(tf.float32, shape=[TRAIN_IMAGES, 480, 360, 3])
    
    #alpha = target_images[:,:,:,3][..., np.newaxis]

    with tf.variable_scope("Gen"):
        gen = UNet(4,3)
        output = tf.sigmoid(gen(input_images))
        g_loss = tf.losses.mean_squared_error(target_images, output)
    with tf.variable_scope("Disc"):
        disc = Discriminator(4)
        d_real = disc(target_images)
        d_fake = disc(output)
        d_loss = tf.reduce_mean(tf.log(d_real) + tf.log(1-d_fake))

    a_loss = g_loss + args.d_coeff * d_loss

    g_loss_summary = tf.summary.scalar("g_loss", g_loss)
    d_loss_summary = tf.summary.scalar("d_loss", d_loss)
    a_loss_summary = tf.summary.scalar("a_loss", a_loss)

    summary_op = tf.summary.merge(
        [g_loss_summary, d_loss_summary, a_loss_summary])

    summary_image = tf.summary.image("result", output)

    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')

    g_optimizer = tf.train.AdadeltaOptimizer(args.lr).minimize(g_loss, global_step=global_step, var_list=g_vars)
    a_optimizer = tf.train.AdadeltaOptimizer(args.lr).minimize(a_loss, global_step=global_step, var_list=g_vars)
    d_optimizer = tf.train.AdadeltaOptimizer(args.lr).minimize(-d_loss, global_step=global_step, var_list=d_vars)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    train_writer = tf.summary.FileWriter(args.logdir + '/train')
    test_writer = tf.summary.FileWriter(args.logdir + '/test')
    saver = tf.train.Saver()
    if args.checkpoint is not None and os.path.exists(os.path.join(args.logdir, 'checkpoint')):
        if args.checkpoint == -1:#latest checkpoint
            saver.restore(sess, tf.train.latest_checkpoint(args.logdir))
        else:#Specified checkpoint
            saver.restore(sess, os.path.join(args.logdir, model_name+".ckpt-"+str(args.checkpoint)))
        print('Model restored to step ' + str(global_step.eval(sess)))


    train_ids = list(train_ids.eval(sess))
    valid_ids = list(valid_ids.eval(sess))

    def load_batch(batch_ids):
        images, targets = [], []
        test_data = np.zeros((TRAIN_IMAGES, 480, 360, 4))
        test_answ = np.zeros((TRAIN_IMAGES, int(480 * 1), int(360 * 1), 3))
        i = 0
        for names in os.listdir(input_path):
            if i < TRAIN_IMAGES:
                random_file = random.choice(os.listdir(input_path))
                name = os.path.basename(random_file.split(".")[0])
                input_filename = os.path.join(input_path, name + '.jpg')
                trimap_filename = os.path.join(trimap_path, name + '.png')
                target_filename = os.path.join(target_path, name + '.png')
                I = misc.imread(input_filename)
                I_depth = misc.imread(trimap_filename)

                if i == 20:
                    print('random_file: '+random_file)

                I = cv2.resize(I, (360, 480));
                I_depth = cv2.resize(I_depth, (360, 480));
                # Stacking the image together with its depth map
                I_temp = np.zeros((I.shape[0], I.shape[1], 4))
                I_temp[:, :, 0:3] = I
                I_temp[:, :, 3] = I_depth
                I = I_temp

               
                # Extracting random patch of width PATCH_WIDTH
               
                I = np.float32(I) / 255.0

                test_data[i, :] = I

                I = misc.imread(target_filename)
                I = cv2.resize(I, (int(360), int(480)));
                
                I = np.float32(I) / 255.0

                test_answ[i, :] = I
                i = i + 1

        return test_data, test_answ


    def test_step(batch_idx, summary_fct):
        batch_range = random.sample(train_ids, args.batch_size)

        images, targets = load_batch(batch_range)

        loss, demo, summary = sess.run([g_loss, summary_image, summary_fct], feed_dict={
            input_images: images,
            target_images: targets,
            })

        test_writer.add_summary(summary, batch_idx)
        test_writer.add_summary(demo, batch_idx)
        print("---------------------------------------------")
        print(output)
        image1_tensor= output[0]
        print(image1_tensor.shape)
        image1_numpy = tf.cast(image1_tensor, tf.uint8)
        print(image1_numpy)
        print('Validation Loss: {:.8f}'.format(loss))

    batch_idx = 0
    while batch_idx < n_iter:
        batch_idx = global_step.eval(sess) * args.batch_size
        print('batch_idx' + str(batch_idx))

        loss_fct = None
        label = None
        optimizers = []
        if batch_idx < g_iter:
            print('Gen train')
            loss_fct = g_loss
            summary_fct = g_loss_summary
            label = 'Gen train'
            optimizers = [g_optimizer]
        elif batch_idx < g_iter+d_iter:
            loss_fct = d_loss
            summary_fct = d_loss_summary
            label = 'Disc train'
            optimizers = [d_optimizer]
            print('Disc train')
        else:
            loss_fct = a_loss
            summary_fct = summary_op
            label = 'Adv train'
            optimizers = [a_optimizer]
            print('Adv train')

        batch_range = random.sample(train_ids, args.batch_size)
        images, targets = load_batch(batch_range)
        print('load batch done')
        loss, summary = sess.run([loss_fct, summary_fct] +  optimizers, feed_dict={
            input_images: images,
            target_images: targets})[0:2]
        print('load batch done 1')
        if batch_idx % train_data_update_freq == 0:
            print('{}: [{}/{} ({:.0f}%)]\tGen Loss: {:.8f}'.format(label, batch_idx, n_iter,
                100. * (batch_idx+1) / n_iter, loss))
            print('load batch done 2')
            train_writer.add_summary(summary, batch_idx)

        if batch_idx % test_data_update_freq == 0:
            print('load batch done 3')
            test_step(batch_idx, summary_fct)

        if batch_idx % sess_save_freq == 0:
            print('Saving model')
            saver.save(sess, os.path.join(args.logdir, model_name+".ckpt"), global_step=batch_idx)

    saver.save(sess, os.path.join(args.logdir, model_name+".ckpt"), global_step=batch_idx)


if __name__ == '__main__':
    args = parse_args()
    main(args)
