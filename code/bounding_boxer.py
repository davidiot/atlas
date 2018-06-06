import numpy as np
import glob
import os
import json

from PIL import Image


def compute_bounding_box(input_path, padding, show_image=False):
    input_image = Image.open(input_path).convert("1")  # convert to black and white
    input_array = np.array(input_image)
    vertical_indices = np.argwhere(input_array.sum(axis=0))[:, 0]
    horizontal_indices = np.argwhere(input_array.sum(axis=1))[:, 0]
    x1 = vertical_indices[0]
    x2 = vertical_indices[-1]
    y1 = horizontal_indices[0]
    y2 = horizontal_indices[-1]
    bbox = (max(x1 - padding, 0),
            max(y1 - padding, 0),
            min(x2 + padding, input_array.shape[0] - 1),
            min(y2 + padding, input_array.shape[1] - 1))
    if show_image:
        input_image.crop(bbox).show()
    return bbox


def generate_canonical_boxes(FLAGS):
    prefix = os.path.join(FLAGS.data_dir, "ATLAS_R1.1")
    mask_slice_paths = glob.glob(os.path.join(prefix, "Site*/**/*/*LesionSmooth*/*.jpg"))

    max_height_path = ''
    max_width_path = ''
    min_height_path = ''
    min_width_path = ''
    max_width = 0
    max_height = 0
    min_width = 150
    min_height = 150

    for slice_path in mask_slice_paths:
        try:
            bbox = compute_bounding_box(slice_path, 0)
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            if width > max_width:
                max_width_path = slice_path
                max_width = width
            elif width < min_width:
                min_width_path = slice_path
                min_width = width
            if height > max_height:
                max_height_path = slice_path
                max_height = height
            elif height < min_height:
                min_height_path = slice_path
                min_height = height
            # base_name = os.path.splitext(slice_path)[0]
            # with open(base_name + "-canonical.json", "w") as fout:
            #     json.dump(compute_bounding_box(slice_path, 3, True), fout)
        except IndexError:
            pass
    print("max_height: ", max_height, " | ", max_height_path)
    print("max_width: ", max_width, " | ", max_width_path)
    print("min_height: ", min_height, " | ", min_height_path)
    print("min_width: ", min_width, " | ", min_width_path)