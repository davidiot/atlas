import numpy as np
import glob
import os
import json

from PIL import Image

# counts: {(10, 10): 4088, (30, 30): 3228, (70, 70): 3049, (150, 150): 1313}
BOX_SHAPES = [(10, 10), (30, 30), (70, 70), (150, 150)]
BOX_LABELS = {(10, 10): 'S', (30, 30): 'M', (70, 70): 'L', (150, 150): 'XL'}


# if the edges of the box are outside of the input image, shift it
def shift_box(input_array, x1, y1, x2, y2):
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
    assert(x1 >= 0)
    assert(y1 >= 0)
    assert(x2 <= input_array.shape[0] - 1)
    assert(y2 <= input_array.shape[1] - 1)
    return int(x1), int(y1), int(x2), int(y2)


def box_width(box):
    return box[2] - box[0]


def box_height(box):
    return box[3] - box[1]


def compute_bounding_box(slice_path, show_image=False):
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
    for shape in BOX_SHAPES:
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


# Given an inner box (the smallest box containing the entire mask)
# and a canonical box, generate random slices that still contain the outer box
def augment_boxes(canonical_box, inner_box, slice_path, num_crops, show_image=False):
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
    return tuple(output)


# Determines the canonical and augmented bounding boxes.
def generate_boxes(FLAGS):
    prefix = os.path.join(FLAGS.data_dir, "ATLAS_R1.1")
    mask_slice_paths = glob.glob(os.path.join(prefix, "Site*/**/*/*LesionSmooth*/*.jpg"))

    shape_counts = {shape: 0 for shape in BOX_SHAPES}
    augment_counts = {shape: 0 for shape in BOX_SHAPES}
    for slice_path in mask_slice_paths:
        try:
            shape, bbox, inner_box = compute_bounding_box(slice_path)
            shape_counts[shape] += 1
            base_name = os.path.splitext(slice_path)[0]
            label = BOX_LABELS[shape]
            with open(base_name + "-canonical-" + label + ".json", "w") as fout:
                json.dump(bbox, fout)
            augmented_boxes = augment_boxes(bbox, inner_box, slice_path, FLAGS.num_crops)
            augment_counts[shape] += len(augmented_boxes)
            with open(base_name + "-augmented-" + label + ".json", "w") as fout:
                json.dump(augmented_boxes, fout)
        except IndexError:
            pass  # There were no pixels in the mask.
    for shape in shape_counts:
        print("Generated ", shape_counts[shape], " canonical boxes of shape ", shape, " (", BOX_LABELS[shape], ")")
    for shape in augment_counts:
        print("Generated ", augment_counts[shape], " total boxes of shape ", shape, " (", BOX_LABELS[shape], ")")
