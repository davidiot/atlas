import numpy as np
from PIL import Image


def compute_bounding_box(input_path, padding, show_image=False):
    input_image = Image.open(input_path).convert("1") # convert to black and white
    input_array = np.array(input_image)
    vertical_indices = np.argwhere(input_array.sum(axis=0))[:, 0]
    horizontal_indices = np.argwhere(input_array.sum(axis=1))[:, 0]
    bbox = (max(vertical_indices[0] - padding, 0),
            max(horizontal_indices[0] - padding, 0),
            min(vertical_indices[-1] + padding, input_array.shape[0] - 1),
            min(horizontal_indices[-1] + padding, input_array.shape[1] - 1))
    if show_image:
        input_image.crop(bbox).show()
    return bbox
