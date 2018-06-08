import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time

import utils
from modules import ConvEncoder, DeconvDecoder, UNet
from atlas_model import ATLASModel
from box_batcher import BoxBatchGenerator


class BoxAtlasModel(ATLASModel):
    def __init__(self, FLAGS):
        """
        Initializes the BoxAtlasModel, which takes bounding boxes as inputs
        instead of entire slices.

        :param FLAGS: A _FlagValuesWrapper object passed in from main.py.
        """
        super().__init__(FLAGS)
        assert(not FLAGS.use_volumetric)

    def get_batch_generator(self, input_paths, target_mask_paths, num_samples=None, flip_images=True):
        return BoxBatchGenerator(input_paths,
                                 target_mask_paths,
                                 self.FLAGS.batch_size,
                                 num_samples=num_samples,
                                 shape=(self.FLAGS.slice_height,
                                        self.FLAGS.slice_width),
                                 shuffle=True,
                                 use_fake_target_masks=self.FLAGS.use_fake_target_masks,
                                 flip_images=flip_images)

    def calculate_dice_coefficient(self,
                                   sess,
                                   input_paths,
                                   target_mask_paths,
                                   dataset,
                                   num_samples=100,
                                   flip_images=True,
                                   plot=False,
                                   print_to_screen=False):
        """
        Calculates the dice coefficient score for a dataset, represented by a
        list of {input_paths} and {target_mask_paths}.

        Inputs:
        - sess: A TensorFlow Session object.
        - input_paths: A list of Python strs that represent pathnames to input
          image files.
        - target_mask_paths: A list of Python strs that represent pathnames to
          target mask files.
        - dataset: A Python str that represents the dataset being tested. Options:
          {train,dev}. Just for logging purposes.
        - num_samples: A Python int that represents the number of samples to test.
          If num_samples=None, then test whole dataset.
        - plot: A Python bool. If True, plots each example to screen.

        Outputs:
        - dice_coefficient: A Python float that represents the average dice
          coefficient across the sampled examples.
        """
        logging.info(f"Calculating dice coefficient for {num_samples} examples "
                     f"from {dataset}...")
        tic = time.time()

        dice_coefficient_total = 0.
        num_examples = 0

        sbg = self.get_batch_generator(input_paths, target_mask_paths, flip_images=flip_images)

        for batch in sbg.get_batch():
            predicted_masks = self.get_predicted_masks_for_batch(sess, batch)

            zipped_masks = zip(predicted_masks,
                               batch.target_masks_batch,
                               batch.input_paths_batch,
                               batch.boxes_batch,
                               batch.target_mask_paths_batch)
            for idx, (predicted_mask,
                      target_mask,
                      input_path,
                      box,
                      target_mask_path) in enumerate(zipped_masks):
                dice_coefficient = utils.dice_coefficient(predicted_mask, target_mask)
                if dice_coefficient >= 0.0:
                    dice_coefficient_total += dice_coefficient
                    num_examples += 1

                    if print_to_screen:
                        # Whee! We predicted at least one lesion pixel!
                        logging.info(f"Dice coefficient of valid example {num_examples}: "
                                     f"{dice_coefficient}")
                    if plot:
                        f, axarr = plt.subplots(1, 2)
                        f.suptitle(input_path + " | " + target_mask_path)
                        axarr[0].imshow(predicted_mask)
                        axarr[0].set_title("Predicted")
                        axarr[1].imshow(target_mask)
                        axarr[1].set_title("Target")
                        examples_dir = os.path.join(self.FLAGS.train_dir, "examples")
                        if not os.path.exists(examples_dir):
                            os.makedirs(examples_dir)
                        f.savefig(os.path.join(examples_dir, str(num_examples).zfill(4)))

                if num_samples != None and num_examples >= num_samples:
                    break

            if num_samples != None and num_examples >= num_samples:
                break

        dice_coefficient_mean = dice_coefficient_total / num_examples

        toc = time.time()
        logging.info(f"Calculating dice coefficient took {toc-tic} sec.")
        return dice_coefficient_mean


class ZeroBoxAtlasModel(BoxAtlasModel):
    def __init__(self, FLAGS):
        """
        Initializes the Zero ATLAS model, which predicts 0 for the entire mask
        no matter what, which performs well when --use_fake_target_masks.

        Inputs:
        - FLAGS: A _FlagValuesWrapper object passed in from main.py.
        """
        super().__init__(FLAGS)

    def build_graph(self):
        """
        Sets {self.logits_op} to a matrix entirely of a small constant.
        """
        # -18.420680734 produces a sigmoid-ce loss of ~10^-8
        c = tf.get_variable(initializer=tf.constant_initializer(-18.420680734),
                            name="c",
                            shape=())
        self.logits_op = tf.ones(shape=[self.FLAGS.batch_size] + self.input_dims,
                                 dtype=tf.float32) * c
        self.predicted_mask_probs_op = tf.sigmoid(self.logits_op,
                                                  name="predicted_mask_probs")
        self.predicted_masks_op = tf.cast(self.predicted_mask_probs_op > 0.5,
                                          tf.uint8,
                                          name="predicted_masks")
