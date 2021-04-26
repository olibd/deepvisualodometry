import glob
import os
import sys
import time
from itertools import compress
from typing import List

import PIL
import numpy
import numpy as np
import pandas as pd
import torch
from PIL.Image import Image
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from stopit import ThreadingTimeout

from Common.Helpers import TracebackSignalTimeout, Geometry
from Logging import CometLogger
from .Interfaces import AbstractSegmentDataset, AbstractSegment, AbstractDataPreprocessor, AbstractSegmenter


class Segment(AbstractSegment):

    def __init__(self, data_series):
        super().__init__(data_series.seq_len)
        self.data_series = data_series

    def get_images(self) -> list:
        image_path_sequence = self.get_images_path()

        image_sequence = []
        for img_path in image_path_sequence:
            image_sequence.append(PIL.Image.open(img_path))
        return image_sequence

    def get_images_path(self) -> list:
        return list(self.data_series.image_path)

    def _get_raw_pose(self):
        return np.asarray(self.data_series.pose)

    def _get_raw_positions(self) -> torch.Tensor:
        return torch.Tensor(self._get_raw_pose()[:, 9:])

    def _get_resetted_attitude(self) -> list:
        raw_attitude = self._get_raw_attitude()
        return self._reset_attitude_origin(raw_attitude)

    def _get_raw_attitude_as_quat(self) -> List[Quaternion]:
        return Geometry.rotation_matrices_to_quaternions(self._get_raw_attitude())

    def _get_raw_attitude(self) -> np.ndarray:
        return self._get_raw_pose()[:, :9]

    def _reset_attitude_origin(self, attitude: np.ndarray) -> list:
        """Reset the sequence's rotation relative to the first frame. Expects angles to be in quaternions"""
        initial_orientation_quat = Geometry.matrix_to_quaternion(attitude[0])
        resetted_attitude = []

        for i, orientation in enumerate(attitude):
            current_orientation_quat = Geometry.matrix_to_quaternion(orientation)
            resetted_oriention_quat = initial_orientation_quat.inverse * current_orientation_quat
            resetted_attitude.append(resetted_oriention_quat)

        return resetted_attitude


class KITTIImageSequenceDataset(AbstractSegmentDataset):
    def data_is_relative(self) -> bool:
        return False

    def __init__(self, dataset_path, resize_mode='crop', new_size=None, img_mean=None, img_std=(1, 1, 1),
                 minus_point_5=False, augment_dataset: bool = False):
        super().__init__(dataset_path,
                         framerate=10,
                         resize_mode=resize_mode,
                         new_size=new_size,
                         img_mean=img_mean,
                         img_std=img_std,
                         minus_point_5=minus_point_5,
                         augment_dataset=augment_dataset)
        self.dataframe: pd.DataFrame = pd.read_pickle(self.dataset_path)
        self.segments = self._load_segments()

    def _load_segments(self) -> list:
        segments = []
        for i in range(0, len(self.dataframe.pose)):
            data = {'seq_len': self.dataframe.seq_len[i], 'image_path': self.dataframe.image_path[i], 'pose': self.dataframe.pose[i]}
            data_series = pd.Series(data)
            segments.append(Segment(data_series))

        return segments


class KITTIImageSequenceDatasetEulerDifferences(KITTIImageSequenceDataset):
    def __init__(self, dataset_segments_path, resize_mode='crop', new_size=None, img_mean=None, img_std=(1, 1, 1),
                 minus_point_5=False, augment_dataset: bool = False):
        super().__init__(dataset_segments_path, resize_mode, new_size, img_mean, img_std, minus_point_5, augment_dataset)

    def __getitem__(self, item: int):
        with ThreadingTimeout(3600.0) as timeout_ctx1:
            try:
                segment, image_sequence = super().__getitem__(item)
            except Exception as e:
                CometLogger.print(str(e))
                raise e
        if not bool(timeout_ctx1):
            CometLogger.fatalprint('Encountered fatal delay while getting the image sequence')

        with ThreadingTimeout(3600.0) as timeout_ctx2:
            pose = self._get_segment_pose(segment)
        if not bool(timeout_ctx2):
            CometLogger.fatalprint('Encountered fatal delay while getting the pose of the sequence')

        return image_sequence, pose

    def _get_segment_pose(self, segment: Segment) -> torch.Tensor:
        position = segment.get_position_differences()
        attitude = segment.get_attitude_differences()
        # att.elements[[1, 2, 3, 0]] reorganizes quaternion elements from scalar first w-x-y-z to scalar last x-y-z-w
        # output is intrinsic Tait-Bryan angles following the y-x'-z''
        attitude = torch.Tensor([np.around(Rotation.from_quat(att.elements[[1, 2, 3, 0]]).as_euler("YXZ"), 7)
                                      for att in attitude])

        return torch.cat([attitude, position], 1)

    def data_is_relative(self) -> bool:
        return True


class KITTIDataSegmenter(AbstractSegmenter):

    def __init__(self, folder_list, pose_dir, image_dir, segments_destination):
        self.folder_list = folder_list
        self.pose_dir = pose_dir
        self.image_dir = image_dir
        self.segments_destination = segments_destination

    def segment(self, seq_len_range, overlap, sample_times=1, pad_y=False, shuffle=False, sort=True):
        """
        Segmentation logic taken from https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/data_helper.py#L15
        @param folder_list:
        @param seq_len_range:
        @param overlap:
        @param sample_times:
        @param pad_y:
        @param shuffle:
        @param sort:
        @return:
        """
        X_path, Y = [], []
        X_len = []
        for folder in self.folder_list:
            start_t = time.time()
            poses = np.load('{}{}.npy'.format(self.pose_dir, folder))  # (n_images, 6)
            fpaths = glob.glob('{}{}/*.png'.format(self.image_dir, folder))
            fpaths.sort()

            #If no seq_len_range is specified, then we assume that the whole video in the folder is 1 segment
            if seq_len_range == None:
                n_frames = len(fpaths)
                seq_len_range = (n_frames,)

            # Fixed seq_len
            if len(seq_len_range) == 1 or seq_len_range[0] == seq_len_range[1]:
                if sample_times > 1:
                    sample_interval = int(np.ceil(seq_len_range[0] / sample_times))
                    start_frames = list(range(0, seq_len_range[0], sample_interval))
                    print('Sample start from frame {}'.format(start_frames))
                else:
                    start_frames = [0]

                for st in start_frames:
                    seq_len = seq_len_range[0]
                    n_frames = len(fpaths) - st
                    jump = seq_len - overlap
                    res = n_frames % seq_len
                    if res != 0:
                        n_frames = n_frames - res
                    x_segs = [fpaths[i:i + seq_len] for i in range(st, n_frames, jump)]
                    y_segs = [poses[i:i + seq_len] for i in range(st, n_frames, jump)]
                    x_len = [len(xs) for xs in x_segs]
                    valid_lengths_indices = [i == seq_len for i in x_len]
                    Y += list(compress(y_segs, valid_lengths_indices))
                    X_path += list(compress(x_segs, valid_lengths_indices))
                    X_len += list(compress(x_len, valid_lengths_indices))
            # Random segment to sequences with diff lengths
            else:
                assert (overlap < min(seq_len_range))
                n_frames = len(fpaths)
                min_len, max_len = seq_len_range[0], seq_len_range[1]
                for i in range(sample_times):
                    start = 0
                    while True:
                        n = np.random.random_integers(min_len, max_len)
                        if start + n < n_frames:
                            x_seg = fpaths[start:start + n]
                            X_path.append(x_seg)
                            if not pad_y:
                                Y.append(poses[start:start + n])
                            else:
                                pad_zero = np.zeros((max_len - n, 12))
                                padded = np.concatenate((poses[start:start + n], pad_zero))
                                Y.append(padded.tolist())
                        else:
                            print('Last %d frames is not used' % (start + n - n_frames))
                            break
                        start += n - overlap
                        X_len.append(len(x_seg))
            print('Folder {} finish in {} sec'.format(folder, time.time() - start_t))

        # Convert to pandas dataframes
        data = {'seq_len': X_len, 'image_path': X_path, 'pose': Y}
        df = pd.DataFrame(data, columns=['seq_len', 'image_path', 'pose'])
        # Shuffle through all videos
        if shuffle:
            df = df.sample(frac=1)
        # Sort dataframe by seq_len
        if sort:
            df = df.sort_values(by=['seq_len'], ascending=False)
        df.to_pickle(self.segments_destination)
        return df


class KITTIDataPreprocessor(AbstractDataPreprocessor):

    def __init__(self, image_dir: str, pose_dir: str, kitti_path: str):
        super().__init__(kitti_path)
        self.image_dir = image_dir
        self.pose_dir = pose_dir

    def clean(self):
        self._clean_unused_images()
        self._create_pose_data()

    def _get_image_path_iterator(self):
        #TODO: Update videos used for mean computation
        train_video = ['00', '02', '08', '09', '06', '04', '10']
        image_path_list = []
        for folder in train_video:
            image_path_list += glob.glob('{}{}/*.png'.format(self.image_dir, folder))

        for img_path in image_path_list:
            yield img_path

    def _clean_unused_images(self):
        """
        logic taken from https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/preprocess.py
        """
        seq_frame = {'00': ['000', '004540'],
                     '01': ['000', '001100'],
                     '02': ['000', '004660'],
                     '03': ['000', '000800'],
                     '04': ['000', '000270'],
                     '05': ['000', '002760'],
                     '06': ['000', '001100'],
                     '07': ['000', '001100'],
                     '08': ['001100', '005170'],
                     '09': ['000', '001590'],
                     '10': ['000', '001200']
                     }
        for dir_id, img_ids in seq_frame.items():
            dir_path = '{}{}/'.format(self.image_dir, dir_id)
            if not os.path.exists(dir_path):
                continue

            print('Cleaning {} directory'.format(dir_id))
            start, end = img_ids
            start, end = int(start), int(end)
            for idx in range(0, start):
                img_name = '{:010d}.png'.format(idx)
                img_path = '{}{}/{}'.format(self.image_dir, dir_id, img_name)
                if os.path.isfile(img_path):
                    os.remove(img_path)
            for idx in range(end + 1, 10000):
                img_name = '{:010d}.png'.format(idx)
                img_path = '{}{}/{}'.format(self.image_dir, dir_id, img_name)
                if os.path.isfile(img_path):
                    os.remove(img_path)

    def _create_pose_data(self):
        """
        logic taken from https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/preprocess.py
        transform poseGT [R|t] to [3x3 flat rotation matrix, x, y, z]
        save as .npy file
        """
        info = {'00': [0, 4540], '01': [0, 1100], '02': [0, 4660], '03': [0, 800], '04': [0, 270], '05': [0, 2760],
                '06': [0, 1100], '07': [0, 1100], '08': [1100, 5170], '09': [0, 1590], '10': [0, 1200]}
        start_t = time.time()
        for video in info.keys():
            fn = '{}{}.txt'.format(self.pose_dir, video)
            print('Transforming {}...'.format(fn))
            with open(fn) as f:
                lines = [line.split('\n')[0] for line in f.readlines()]
                poses = [self._Rt_to_pose([float(value) for value in l.split(' ')]) for l in
                         lines]
                poses = np.array(poses)
                base_fn = os.path.splitext(fn)[0]
                np.save(base_fn + '.npy', poses)
                print('Video {}: shape={}'.format(video, poses.shape))
        print('elapsed time = {}'.format(time.time() - start_t))

    def _Rt_to_pose(self, Rt):
        """
        Modified from https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/helper.py#L14
        # Ground truth pose is present as [R | t]
        # R: Rotation Matrix (3x3), t: translation vector (x,y,z)
        @param Rt:
        @return: A pose [R | t] where R is a flattened 3x3 rotation matrix and t is an x,y,z translation vector
        """

        Rt = np.reshape(np.array(Rt), (3, 4))
        t = Rt[:, -1]
        R = Rt[:, :3]

        assert (self._isRotationMatrix(R))

        pose_12 = np.concatenate((R.flatten(), t))
        assert (pose_12.shape == (12,))
        return pose_12

    @staticmethod
    def _isRotationMatrix(R):
        """
        taken from https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/helper.py#L7
        @param R:
        @return:
        """
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6