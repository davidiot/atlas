import numpy as np
import glob
import os
import json

from PIL import Image

# counts: {(10, 10): 4088, (30, 30): 3228, (70, 70): 3049, (150, 150): 1313}
BOX_SHAPES = [(10, 10), (30, 30), (70, 70), (150, 150)]
BOX_LABELS = {(10, 10): 'S', (30, 30): 'M', (70, 70): 'L', (150, 150): 'XL'}


def compute_bounding_box(input_path, show_image=False):
    input_image = Image.open(input_path).convert("1")  # convert to black and white
    input_array = np.array(input_image)
    vertical_indices = np.argwhere(input_array.sum(axis=0))[:, 0]
    horizontal_indices = np.argwhere(input_array.sum(axis=1))[:, 0]

    x1 = vertical_indices[0]
    x2 = vertical_indices[-1] + 1
    y1 = horizontal_indices[0]
    y2 = horizontal_indices[-1] + 1

    # Determine the smallest box shape that fits the mask
    height = x2 - x1
    width = y2 - y1
    output_box_shape = input_array.shape
    for shape in BOX_SHAPES:
        if width <= shape[0] + 2 and height <= shape[1] + 2:
            output_box_shape = shape
            break

    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)

    # if the edges of the box are outside of the input image, shift it
    output_x1 = int(x_center - output_box_shape[0] / 2)
    output_x2 = int(x_center + output_box_shape[0] / 2)
    output_y1 = int(y_center - output_box_shape[1] / 2)
    output_y2 = int(y_center + output_box_shape[1] / 2)
    if output_x1 < 0:
        diff = 0 - output_x1
        output_x1 += diff
        output_x2 += diff
    if output_y1 < 0:
        diff = 0 - output_y1
        output_y1 += diff
        output_y2 += diff
    if output_x2 > input_array.shape[0] - 1:
        diff = output_x2 - input_array.shape[0] + 1
        output_x1 -= diff
        output_x2 -= diff
    if output_y2 > input_array.shape[1] - 1:
        diff = output_y2 - input_array.shape[1] + 1
        output_y1 -= diff
        output_y2 -= diff

    bbox = (output_x1, output_y1, output_x2, output_y2)
    assert(output_x2 - output_x1 == output_box_shape[0])
    assert(output_y2 - output_y1 == output_box_shape[1])
    assert(output_x1 >= 0)
    assert(output_y1 >= 0)
    assert(output_x2 <= input_array.shape[0] - 1)
    assert(output_y2 <= input_array.shape[1] - 1)
    if show_image:
        input_image.crop(bbox).show()
    return output_box_shape, bbox


def generate_canonical_boxes(FLAGS):
    prefix = os.path.join(FLAGS.data_dir, "ATLAS_R1.1")
    mask_slice_paths = glob.glob(os.path.join(prefix, "Site*/**/*/*LesionSmooth*/*.jpg"))

    shape_counts = {shape: 0 for shape in BOX_SHAPES}
    for slice_path in mask_slice_paths:
        try:
            shape, bbox = compute_bounding_box(slice_path)
            shape_counts[shape] += 1
            base_name = os.path.splitext(slice_path)[0]
            label = BOX_LABELS[shape]
            with open(base_name + "-canonical-" + label + ".json", "w") as fout:
                json.dump(bbox, fout)
        except IndexError:
            pass  # There were no pixels in the mask.
    for shape in shape_counts:
        print("Generated ", shape_counts[shape], " boxes of shape ", shape, " (", BOX_LABELS[shape], ")")
