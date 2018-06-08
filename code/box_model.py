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
