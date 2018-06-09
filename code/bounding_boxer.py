import numpy as np
import glob
import os
import json

from PIL import Image

# Generated  3416  canonical boxes of shape  (12, 12)  ( S )
# Generated  3690  canonical boxes of shape  (32, 32)  ( M )
# Generated  2644  canonical boxes of shape  (64, 64)  ( L )
# Generated  1928  canonical boxes of shape  (160, 160)  ( XL )
# Generated  501401  total boxes, including  159106  positive examples of shape  (12, 12)  ( S )
# Generated  681839  total boxes, including  338216  positive examples of shape  (32, 32)  ( M )
# Generated  613988  total boxes, including  275666  positive examples of shape  (64, 64)  ( L )
# Generated  486804  total boxes, including  153040  positive examples of shape  (160, 160)  ( XL )
BOX_LABELS = {(12, 12): 'S', (32, 32): 'M', (64, 64): 'L', (160, 160): 'XL'}


def box_width(box):
    """ compute the width of a box

    :param box: box coordinates in form (x1, y1, x2, y2)
    :return: box width
    """
    return box[2] - box[0]


def box_height(box):
    """ compute the height of a box

    :param box: box coordinates in form (x1, y1, x2, y2)
    :return: box height
    """
    return box[3] - box[1]


def shift_box(input_array, x1, y1, x2, y2):
    """ if the edges of the box are outside of the original image, shift it

    :param input_array: array of the original image
    :param x1: box coordinate, first corner's x value
    :param y1: box coordinate, first corner's y value
    :param x2: box coordinate, second corner's x value
    :param y2: box coordinate, second corner's y value
    :return: shifted box coordinates
    """
    if x1 < 0:
        diff = 0 - x1
        x1 += diff
        x2 += diff
    if y1 < 0:
        diff = 0 - y1
        y1 += diff
        y2 += diff
    if x2 > input_array.shape[0] - 1:
        diff = x2 - input_array.shape[0] + 1
        x1 -= diff
        x2 -= diff
    if y2 > input_array.shape[1] - 1:
        diff = y2 - input_array.shape[1] + 1
        y1 -= diff
        y2 -= diff
    assert (x1 >= 0)
    assert (y1 >= 0)
    assert (x2 <= input_array.shape[0] - 1)
    assert (y2 <= input_array.shape[1] - 1)
    return int(x1), int(y1), int(x2), int(y2)


def compute_bounding_box(slice_path, show_image=False):
    """
    Computes the canonical bounding box (a box with the smallest possible shape from
    BOX_LABELS that contains the entire mask), and the inner box (the smallest box of
    any shape that contains the entire mask, with one pixel of padding on all sides)

    :param slice_path: path to the mask slice image
    :param show_image: show the canonical bounding box if true
    :return: shape of canonical box, canonical box, inner box tuple
    """
    input_image = Image.open(slice_path).convert("1")  # convert to black and white
    input_array = np.array(input_image)
    vertical_indices = np.argwhere(input_array.sum(axis=0))[:, 0]
    horizontal_indices = np.argwhere(input_array.sum(axis=1))[:, 0]

    x1 = vertical_indices[0]
    x2 = vertical_indices[-1] + 1
    y1 = horizontal_indices[0]
    y2 = horizontal_indices[-1] + 1

    inner_box = (x1, y1, x2, y2)

    # Determine the smallest box shape that fits the mask
    output_box_shape = input_array.shape
    for shape in BOX_LABELS:
        if box_width(inner_box) + 2 <= shape[0] and box_height(inner_box) + 2 <= shape[1]:
            output_box_shape = shape
            break

    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)

    output_x1 = int(x_center - output_box_shape[0] / 2)
    output_x2 = int(x_center + output_box_shape[0] / 2)
    output_y1 = int(y_center - output_box_shape[1] / 2)
    output_y2 = int(y_center + output_box_shape[1] / 2)
    output_x1, output_y1, output_x2, output_y2 = shift_box(input_array, output_x1, output_y1, output_x2, output_y2)

    bbox = (output_x1, output_y1, output_x2, output_y2)
    assert(box_width(bbox) == output_box_shape[0])
    assert(box_height(bbox) == output_box_shape[1])
    if show_image:
        input_image.crop(bbox).show()
    return output_box_shape, bbox, inner_box


def augment_boxes(canonical_box, inner_box, slice_path, num_crops, show_image=False):
    """
    Given an inner box (the smallest box containing the entire mask) and a canonical box,
    generate random crops with the same size as the canonical box that still contain the
    entire inner box.

    :param canonical_box: the canonical box used to train the first neural net
    :param inner_box: the smallest box containing the entire mask
    :param slice_path: path to the mask slice image
    :param num_crops: how many crops to attempt to generate; this may produce duplicates
    :param show_image: show the augmented boxes if true
    :return: a set containing the canonical box and all augmented boxes
    """
    input_image = Image.open(slice_path).convert("1")  # convert to black and white
    input_array = np.array(input_image)
    output = set([canonical_box])
    wiggle_x = (box_width(canonical_box) - box_width(inner_box)) / 2
    wiggle_y = (box_height(canonical_box) - box_height(inner_box)) / 2
    for _ in range(num_crops):
        x_shift = np.random.random_integers(-wiggle_x, wiggle_x)
        y_shift = np.random.random_integers(-wiggle_y, wiggle_y)
        shifted_box = shift_box(
            input_array,
            canonical_box[0] + x_shift,
            canonical_box[1] + y_shift,
            canonical_box[2] + x_shift,
            canonical_box[3] + y_shift
        )
        output.add(shifted_box)
        assert (box_width(canonical_box) == box_width(shifted_box))
        assert (box_height(canonical_box) == box_height(shifted_box))
    if show_image:
        for box in output:
            input_image.crop(box).show()
    return output


def generate_additional_examples(shape, slice_path, num_examples, show_image=False):
    """ randomly crops the image into boxes of the given shape

    :param shape: shape of boxes to create
    :param slice_path: path to mask slice
    :param num_examples: number of crops to make
    :param show_image: shows the crops if true
    :return: a set containing the additional examples
    """
    input_image = Image.open(slice_path).convert("1")  # convert to black and white
    input_array = np.array(input_image)
    image_width = input_array.shape[0]
    image_height = input_array.shape[1]
    example_width = shape[0]
    example_height = shape[1]
    output = set()
    wiggle_x = (image_width - example_width) / 2
    wiggle_y = (image_height - example_height) / 2
    for _ in range(num_examples):
        x_shift = np.random.random_integers(0, wiggle_x)
        y_shift = np.random.random_integers(0, wiggle_y)
        shifted_box = shift_box(
            input_array,
            x_shift,
            y_shift,
            x_shift + example_width,
            y_shift + example_height
        )
        output.add(shifted_box)
        assert (example_width == box_width(shifted_box))
        assert (example_height == box_height(shifted_box))
    if show_image:
        for box in output:
            input_image.crop(box).show()
    return output


def generate_boxes(FLAGS):
    """ Determines the canonical and augmented bounding boxes.

    Saves the canonical boxes in json files ending in -canonical-{SHAPE}.json
    Saves the augmented boxes in json files ending in -augmented-{SHAPE}.json

    :param FLAGS: input flags
    :return:
    """
    prefix = os.path.join(FLAGS.data_dir, "ATLAS_R1.1")
    mask_slice_paths = glob.glob(os.path.join(prefix, "Site*/**/*/*LesionSmooth*/*.jpg"))

    shape_counts = {shape: 0 for shape in BOX_LABELS}
    example_counts = {shape: 0 for shape in BOX_LABELS}
    augment_counts = {shape: 0 for shape in BOX_LABELS}
    for slice_path in mask_slice_paths:
        base_name = os.path.splitext(slice_path)[0]
        try:
            shape, bbox, inner_box = compute_bounding_box(slice_path)
            shape_counts[shape] += 1
            label = BOX_LABELS[shape]
            with open(base_name + "-canonical-" + label + ".json", "w") as fout:
                json.dump(bbox, fout)
            positive_examples = augment_boxes(bbox, inner_box, slice_path, FLAGS.num_crops)
            random_examples = generate_additional_examples(shape, slice_path, FLAGS.num_additional_samples)
            augmented_boxes = tuple(positive_examples | random_examples)
            augment_counts[shape] += len(positive_examples)
            example_counts[shape] += len(augmented_boxes)
            with open(base_name + "-augmented-" + label + ".json", "w") as fout:
                json.dump(augmented_boxes, fout)
        except IndexError:  # There were no pixels in the mask, but we can still take negative examples.
            for shape in BOX_LABELS:
                label = BOX_LABELS[shape]
                random_examples = tuple(
                    generate_additional_examples(shape, slice_path, FLAGS.num_additional_samples))
                example_counts[shape] += len(random_examples)
                with open(base_name + "-augmented-" + label + ".json", "w") as fout:
                    json.dump(random_examples, fout)

    for shape in shape_counts:
        print("Generated ", shape_counts[shape], " canonical boxes of shape ", shape, " (", BOX_LABELS[shape], ")")
    for shape in augment_counts:
        print("Generated ", example_counts[shape], " total boxes, including ", augment_counts[shape],
              " positive examples of shape ", shape, " (", BOX_LABELS[shape], ")")
