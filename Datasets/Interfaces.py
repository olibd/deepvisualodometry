import bisect
import hashlib
import math
import os
import pickle
import shutil
import sys
import warnings
from abc import ABC, abstractmethod
from typing import List

import numpy
import torch
from PIL import Image
from pyquaternion import Quaternion
from stopit import SignalTimeout, ThreadingTimeout
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, RandomApply

from Common.DiffAugment_pytorch import DiffAugmentModule

from Logging import CometLogger


class AbstractSegment(ABC):
    def __init__(self, segment_length: int):
        self.segment_length: int = segment_length

    def get_position_differences(self) -> torch.Tensor:
        """
        Returns the position difference between each consecutive poses,
        this is useful in the case the input to the network are two consecutive frames
        """
        position = self.get_positions()
        position[1:] = position[1:] - position[0:-1]
        return position[1:]

    def get_attitude_differences(self) -> list:
        attitude = self._get_resetted_attitude()

        if not all(isinstance(att, Quaternion) for att in attitude):
            raise TypeError('Not all objects are of type Quaternion')

        attitude_difference = []
        last_frame_rotation = None
        for orientation in attitude:
            if last_frame_rotation is not None:
                current_orientation_quat = orientation
                att_diff = current_orientation_quat * last_frame_rotation.inverse
                last_frame_rotation = current_orientation_quat
                attitude_difference.append(att_diff)
            else:
                last_frame_rotation = orientation
        return attitude_difference

    @abstractmethod
    def get_images(self) -> list:
        pass

    @abstractmethod
    def get_images_path(self) -> list:
        pass

    def get_positions(self) -> torch.Tensor:
        """Return position as tensor"""
        position = self._get_raw_positions()
        initial_orientation = self._get_raw_attitude_as_quat()[0]
        initial_orientation_inverse = initial_orientation.inverse
        position = self._reset_position_origin(position)
        self._remap_translation_axes(initial_orientation_inverse, position)
        return position

    def get_attitudes(self) -> list:
        return self._get_resetted_attitude()

    def _get_resetted_attitude(self) -> list:
        raw_attitude = self._get_raw_attitude_as_quat()
        return self._reset_attitude_origin(raw_attitude)

    def _reset_position_origin(self, position: torch.Tensor) -> torch.Tensor:
        """Reset the sequence's position relative to the first frame"""
        return position - position[0]

    def _remap_translation_axes(self, rotation: Quaternion, translations: torch.Tensor):
        rotation_matrix = rotation.rotation_matrix
        for location in translations:
            location[:] = torch.Tensor(rotation_matrix.dot(location.numpy()))

    @abstractmethod
    def _get_raw_positions(self):
        pass

    @abstractmethod
    def _get_raw_attitude_as_quat(self) -> List[Quaternion]:
        pass

    @abstractmethod
    def _get_raw_attitude(self):
        pass

    @abstractmethod
    def _reset_attitude_origin(self, raw_attitude):
        pass

    def __len__(self):
        return self.segment_length

    def __hash__(self):
        return hashlib.sha1("".join(self.get_images_path()).encode("ascii")).hexdigest()

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class AbstractSegmentDataset(Dataset, ABC):
    def __init__(self, dataset_path: str, framerate:int, resize_mode='crop', new_size=None, img_mean: float = None,
                 img_std: float = None, minus_point_5: bool = False, augment_dataset: bool = False):
        self.augment_dataset = augment_dataset
        self.segments: list = []
        self.dataset_path: str = dataset_path
        self.framerate: int = framerate

        # Transforms
        transform_ops = []
        if resize_mode == 'crop':
            transform_ops.append(transforms.CenterCrop((new_size[1], new_size[0])))
        elif resize_mode == 'rescale':
            transform_ops.append(transforms.Resize((new_size[1], new_size[0])))
        transform_ops.append(transforms.ToTensor())
        self.transformer: Compose = transforms.Compose(transform_ops)

        self.minus_point_5: int = minus_point_5
        self.img_mean: float = img_mean
        self.img_std: float = img_std
        self.normalizer = transforms.Normalize(mean=self.img_mean, std=self.img_std)

        if os.path.isfile(self.dataset_path):
            self.dataset_directory = os.path.dirname(self.dataset_path)
        else:
            self.dataset_directory = self.dataset_path

    def get_segments(self) -> list:
        return self.segments

    @abstractmethod
    def data_is_relative(self) -> bool:
        pass

    def __getitem__(self, item: int):
        """
        :param item:
        :return: segment: Segment, image_sequence: torch.FloatTensor
        """
        segment = self._get_segment(item)
        image_sequence = self._load_image_sequence(segment)

        return segment, image_sequence

    def _load_image_sequence(self, segment: AbstractSegment) -> torch.Tensor:
        cache_directory = self.dataset_directory + "/segment_image_tensor_cache"

        self._create_cache_dir(cache_directory)

        try:
            with ThreadingTimeout(2.0) as timeout_ctx1:
                images = torch.load("{}/{}.pkl".format(cache_directory, segment.__hash__()))

            if not bool(timeout_ctx1):
                CometLogger.print('Took too long when loading a cache image. '
                                  'We will load the image directly form the dataset instead.')
                raise Exception()
        except:
            image_sequence = []

            with ThreadingTimeout(3600.0) as timeout_ctx2:
                for img_as_img in segment.get_images():
                    img_as_tensor = self.transformer(img_as_img)
                    if self.minus_point_5:
                        img_as_tensor = img_as_tensor - 0.5  # from [0, 1] -> [-0.5, 0.5]
                    img_as_tensor = self.normalizer(img_as_tensor)
                    img_as_tensor = img_as_tensor.unsqueeze(0)
                    image_sequence.append(img_as_tensor)
                images = torch.cat(image_sequence, 0)
            if not bool(timeout_ctx2):
                CometLogger.fatalprint('Encountered fatal delay when reading the uncached images from the dataset')

            free = -1
            try:
                with ThreadingTimeout(2.0) as timeout_ctx3:
                    _, _, free = shutil.disk_usage(cache_directory)
                if not bool(timeout_ctx3):
                    CometLogger.print('Took too long to measure disk space. Skipping caching.')

            except Exception as e:
                print("Warning: unable to cache the segment's image tensor, there was an error while getting "
                      "disk usage: {}".format(e), file=sys.stderr)

            if free == -1:
                pass
            elif free // (2**30) > 1:
                try:
                    with ThreadingTimeout(5.0) as timeout_ctx4:
                        torch.save(images, "{}/{}.pkl".format(cache_directory, segment.__hash__()))

                    if not bool(timeout_ctx4):
                        CometLogger.print('Took too long when saving to cache folder. Deadlock possible. Skipping caching.')

                except Exception as e:
                    print("Warning: unable to cache the segment's image tensor: {}".format(e), file=sys.stderr)
            else:
                pass

        if self.augment_dataset:
            images = self._augment_image_sequence(images)

        return images

    def _augment_image_sequence(self, images):
        random_apply_random_cutout = RandomApply([DiffAugmentModule("cutout")])
        random_apply_color_jitters = RandomApply([DiffAugmentModule("color")])
        images = random_apply_random_cutout(images)
        images = random_apply_color_jitters(images)
        return images

    def _create_cache_dir(self, cache_directory):
        if not os.path.isdir(cache_directory):
            try:
                os.mkdir(cache_directory)
            except FileExistsError as e:
                # If the directory was created right before we try to create it, the its because an other
                # thread got to it first, so lets just move on.
                pass

    def _get_segment(self, index: int):
        return self.segments[index]

    def __len__(self):
        return len(self.segments)

    def __str__(self):
        return self.__class__.__name__

    def __add__(self, other):
        return _MultiDataset([self, other])

    def __del__(self):
        self.clean_dataset_image_cache()

    def clean_dataset_image_cache(self):
        cache_directory = self.dataset_directory + "/segment_image_tensor_cache/"
        if os.path.isdir(cache_directory):
            try:
                shutil.rmtree(cache_directory)
            except IOError as e:
                # If the directory was deleted right before we try to delete it, the its because an other
                # thread got to it first, so lets just move on.
                pass


class _MultiDataset(AbstractSegmentDataset):
    """
    Modified Pytorch's ConcatDataset. Please do not manually instantiate.
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    def data_is_relative(self) -> bool:
        return self.relative_data

    def get_segments(self) -> list:
        all_segments = []
        for dataset in self.datasets:
            all_segments.extend(dataset.get_segments())

        return all_segments

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)

        for i, dataset in enumerate(self.datasets):
            if i == 0:
                self.relative_data = dataset.data_is_relative()
            else:
                assert self.relative_data == dataset.data_is_relative(), \
                    "All concatenated dataset should have the same type of data " \
                    "(ie. All data should be absolute or relative)"

        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

    def __del__(self):
        pass

class AbstractDataPreprocessor(ABC):

    def __init__(self, mean_std_destination_folder: str):
        self.mean_std_destination_folder: str = mean_std_destination_folder

    @abstractmethod
    def clean(self):
        pass

    def compute_dataset_image_mean_std(self, minus_point_5: bool = False) -> tuple:
        """
        loop through the dataset's images to compute the means and std devs for each channels
        @param minus_point_5:
        @return: tuple of composed of the means' file path and std devs' file path
        """
        dataset_image_means = {"mean_np": [0, 0, 0], "mean_tensor": [0, 0, 0]}
        dataset_image_stds = {"std_np": [0, 0, 0], "std_tensor": [0, 0, 0]}
        pixel_count = 0

        pixel_count = self._compute_mean(dataset_image_means, minus_point_5, pixel_count)
        self._compute_std(dataset_image_means, dataset_image_stds, minus_point_5,
                          pixel_count)

        mean_file_path = '{}/Means.pkl'.format(self.mean_std_destination_folder)
        std_file_path = '{}/StandardDevs.pkl'.format(self.mean_std_destination_folder)
        mean_file = open(mean_file_path, 'wb+')
        std_dev_file = open(std_file_path, 'wb+')

        pickle.dump(dataset_image_means, mean_file)
        pickle.dump(dataset_image_stds, std_dev_file)

        return mean_file_path, std_file_path

    def _compute_std(self, dataset_image_means, dataset_image_stds, minus_point_5,
                     pixel_count):

        image_iterator = self._get_image_path_iterator()
        for path in image_iterator:
            img_as_img = Image.open(path)
            to_tensor = transforms.ToTensor()
            img_as_tensor = to_tensor(img_as_img)

            if minus_point_5:
                img_as_tensor = img_as_tensor - 0.5

            img_as_np = numpy.array(img_as_img)
            img_as_np = numpy.rollaxis(img_as_np, 2, 0)

            for channel in range(3):
                tmp_tensor = (img_as_tensor[channel] - dataset_image_means["mean_tensor"][channel]) ** 2
                dataset_image_stds["std_tensor"][channel] += float(torch.sum(tmp_tensor))
                tmp_array = (img_as_np[channel] - dataset_image_means["mean_np"][channel]) ** 2
                dataset_image_stds["std_np"][channel] += float(numpy.sum(tmp_array))

        dataset_image_stds["std_tensor"] = \
            [math.sqrt(channel_values / pixel_count) for channel_values in dataset_image_stds["std_tensor"]]
        dataset_image_stds["std_np"] = \
            [math.sqrt(channel_values / pixel_count) for channel_values in dataset_image_stds["std_np"]]

    def _compute_mean(self, dataset_image_means, minus_point_5, pixel_count):

        image_iterator = self._get_image_path_iterator()
        for path in image_iterator:
            img_as_img = Image.open(path)
            to_tensor = transforms.ToTensor()
            img_as_tensor = to_tensor(img_as_img)

            if minus_point_5:
                img_as_tensor = img_as_tensor - 0.5

            img_as_np = numpy.array(img_as_img)
            img_as_np = numpy.rollaxis(img_as_np, 2, 0)
            pixel_count += img_as_np.shape[1] * img_as_np.shape[2]

            for channel in range(3):
                dataset_image_means["mean_tensor"][channel] \
                    += float(torch.sum(img_as_tensor[channel]))
                dataset_image_means["mean_np"][channel] \
                    += float(numpy.sum(img_as_np[channel]))

        dataset_image_means["mean_tensor"] = \
            [channel_values / pixel_count for channel_values in dataset_image_means["mean_tensor"]]
        dataset_image_means["mean_np"] = \
            [channel_values / pixel_count for channel_values in dataset_image_means["mean_np"]]

        return pixel_count

    @abstractmethod
    def _get_image_path_iterator(self):
        pass


class AbstractSegmenter:
    pass
