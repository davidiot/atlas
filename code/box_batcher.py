import numpy as np
import random
import json
import os

from PIL import Image
from bounding_boxer import BOX_LABELS


class BoxBatch(object):
    def __init__(self,
                 inputs_batch,
                 target_masks_batch,
                 input_paths_batch,
                 boxes_batch,
                 target_mask_paths_batch):
        """ Initializes a boxes batch

        :param inputs_batch: A numpy array with shape batch_size by input_dims that
        represents a batch of inputs.
        :param target_masks_batch: A numpy array with shape batch_size by input_dims
        that represents a batch of target masks.
        :param input_paths_batch: the input paths
        :param boxes_batch: the boxes used to crop the input and mask
        :param target_mask_paths_batch: The mask paths corresponding to the inputs

        """
        assert(len(
            inputs_batch) == len(
            target_masks_batch) == len(
            inputs_batch) == len(
            target_masks_batch) == len(
            boxes_batch))
        self.inputs_batch = inputs_batch
        self.target_masks_batch = target_masks_batch
        self.input_paths_batch = input_paths_batch
        self.boxes_batch = boxes_batch
        self.target_mask_paths_batch = target_mask_paths_batch
        self.batch_size = len(self.boxes_batch)


class BoxBatchGenerator():
    def __init__(self,
                 input_path_lists,
                 target_mask_path_lists,
                 batch_size,
                 max_num_refill_batches=1000,
                 num_samples=None,
                 shape=(197, 233),
                 shuffle=True,
                 use_fake_target_masks=False,
                 flip_images=False):
        """
        Initalizes a BoxBatchGenerator
        :param input_path_lists: A list of lists of Python strs that represent paths
        to input image files.
        :param target_mask_path_lists:A list of list of lists of Python strs that
        represent paths to target mask files.
        :param batch_size: A Python int that represents the batch size.
        :param max_num_refill_batches: A Python int that represents the maximum number
        of batches of slices to refill at a time.
        :param num_samples: A Python int or None. If None, then uses the entire
        {input_path_lists} and {target_mask_path_lists}.
        :param shape: A Python tuple of ints that represents the shape of the bounding box.
        Note that it must be in bounding_box.BOX_LABELS.
        :param shuffle: A Python bool that represents whether to shuffle the batches
        from the original order specified by {input_path_lists} and
        {target_mask_path_lists}.
        :param use_fake_target_masks: A Python bool that represents whether to use
        fake target masks or not. If True, then {target_mask_path_lists} is
        ignored and all masks are all 0s. This option might be useful to sanity
        check new models before training on the real dataset.
        """
        self._flip_images = flip_images
        self._input_paths = []
        self._boxes = []
        self._target_mask_paths = []
        assert(shape in BOX_LABELS)
        label = BOX_LABELS[shape]
        zipped_path_lists = zip(input_path_lists, target_mask_path_lists)
        for input_path_list, target_mask_path_list in zipped_path_lists:
            assert(len(input_path_list) == 1)
            input_path = input_path_list[0]
            assert(len(target_mask_path_list) == 1)
            mask_paths = target_mask_path_list[0]
            for mask_path in mask_paths:
                base_name = os.path.splitext(mask_path)[0]
                try:
                    with open(base_name + "-augmented-" + label + ".json") as f:
                        boxes = json.load(f)
                        for box in boxes:
                            for _ in range(2 if self._flip_images else 1):  # odds correspond to flipped images
                                self._boxes.append(box)
                                self._input_paths.append(input_path)
                                self._target_mask_paths.append(mask_path)
                except FileNotFoundError:
                    pass

        assert(len(self._boxes) == len(self._input_paths) == len(self._target_mask_paths))
        # print("loaded", len(self._boxes), "examples into a BoxBatchGenerator.")

        self._batch_size = batch_size
        self._batches = []
        self._max_num_refill_batches = max_num_refill_batches
        self._num_samples = num_samples
        if self._num_samples is not None:
            self._input_paths = self._input_paths[:self._num_samples]
            self._boxes = self._boxes[:self._num_samples]
            self._target_mask_paths = self._target_mask_paths[:self._num_samples]
        self._pointer = 0
        self._order = list(range(len(self._boxes)))

        # When the batch_size does not even divide the number of input paths,
        # fill the last batch with randomly selected paths
        num_others = self._batch_size - (len(self._order) % self._batch_size)
        self._order += random.choices(self._order, k=num_others)

        self._shape = shape
        self._use_fake_target_masks = use_fake_target_masks
        if shuffle:
            random.shuffle(self._order)

    def get_num_batches(self):
        """
        Returns the number of batches.
        """
        # The -1 then +1 accounts for the remainder batch.
        return int((len(self._boxes) - 1) / self._batch_size) + 1

    def get_batch(self):
        """
        Returns a generator object that yields batches, the last of which will be
        partial.
        """
        while True:
            if len(self._batches) == 0:  # adds more batches
                self.refill_batches()
            if len(self._batches) == 0:
                break

            # Pops the next batch, a tuple of four items; the first two are numpy
            # arrays {inputs_batch} and {target_mask_batch} of batch_size by
            # input_dims, the last three are tuples of box locations and paths.
            batch = self._batches.pop(0)

            # Wraps the numpy arrays into a Batch object
            batch = BoxBatch(*batch)

            yield batch

    def refill_batches(self):
        """ refills the batches

        :return:
        """
        if self._pointer >= len(self._boxes):
            return

        examples = []  # A Python list of (input, target_mask) tuples

        # {start_idx} and {end_idx} are values like 2000 and 3000
        # If shuffle=True, then {self._order} is a list like [56, 720, 12, ...]
        # {path_indices} is the sublist of {self._order} that represents the
        #   current batch; in other words, the current batch of inputs will be:
        #   [self._input_paths[path_indices[0]],
        #    self._input_paths[path_indices[1]],
        #    self._input_paths[path_indices[2]],
        #    ...]
        # {input_paths}, {boxes}, and {target_mask_paths} are boxes and paths corresponding
        #   to the indices given by {path_indices}
        start_idx, end_idx = self._pointer, self._pointer + self._max_num_refill_batches
        path_indices = self._order[start_idx:end_idx]
        input_paths = [
            self._input_paths[path_idx] for path_idx in path_indices]
        boxes = [
            self._boxes[path_idx] for path_idx in path_indices]
        target_mask_paths = [
            self._target_mask_paths[path_idx] for path_idx in path_indices]

        assert(len(boxes) == len(input_paths) == len(target_mask_paths))

        zipped_path_lists = zip(path_indices, input_paths, boxes, target_mask_paths)

        # Updates self._pointer for the next call to {self.refill_batches}
        self._pointer += self._max_num_refill_batches

        for path_index, input_path, box, target_mask_path in zipped_path_lists:
            cropped_input = Image.open(input_path).convert("L").crop(box)
            if self._flip_images and path_index % 2 == 1:
                cropped_input = cropped_input.transpose(Image.FLIP_LEFT_RIGHT)
            regularized_input = np.asarray(cropped_input) / 255.0
            if self._use_fake_target_masks:
                examples.append((
                    regularized_input,
                    np.zeros(self._shape),
                    input_path,
                    box,
                    "fake_target_mask"
                ))
            else:
                cropped_mask = Image.open(target_mask_path).convert("L").crop(box)
                if self._flip_images and path_index % 2 == 1:
                    cropped_mask = cropped_mask.transpose(Image.FLIP_LEFT_RIGHT)
                regularized_mask = np.asarray(cropped_mask) / 255.0
                target_mask = np.minimum(regularized_mask, 1.0)

                examples.append((
                    regularized_input,
                    target_mask,
                    input_path,
                    box,
                    target_mask_path
                ))
            if len(examples) >= self._batch_size * self._max_num_refill_batches:
                break

        for batch_start_idx in range(0, len(examples), self._batch_size):
            (inputs_batch,
             target_masks_batch,
             input_paths_batch,
             boxes_batch,
             target_mask_paths_batch) = zip(
                *examples[batch_start_idx:batch_start_idx + self._batch_size])

            self._batches.append((
                np.asarray(inputs_batch),
                np.asarray(target_masks_batch),
                input_paths_batch,
                boxes_batch,
                target_mask_paths_batch))
