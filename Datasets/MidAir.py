import os
import zipfile
from typing import Tuple, List

import h5py
import numpy
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pyquaternion import Quaternion

from Common.Helpers import Geometry
from Exception.Exceptions import InvalidSizeException
from .Interfaces import AbstractSegmentDataset, AbstractSegment, AbstractDataPreprocessor, AbstractSegmenter


class TrajectorySegmenter(AbstractSegmenter):
    def segment_trajectory(self, overlap, serialization_destination, sequence_length_range, trajectory_name,
                           trajectory_length):
        if sequence_length_range is None:
            start_frames = [0]
            sequence_lengths = trajectory_length
            sequence = {"sequence_length": sequence_lengths, "start_frame_index": start_frames}
            dataframe = pd.DataFrame(sequence)

            if "trajectory_segments" in serialization_destination:
                del serialization_destination["trajectory_segments"]

            serialization_destination.create_dataset('trajectory_segments', data=dataframe.to_numpy())

        elif len(sequence_length_range) == 1:
            start_frames = list(
                range(0, trajectory_length - sequence_length_range[0], sequence_length_range[0] - overlap))

            dropped_frames = (trajectory_length - 1) - (start_frames[-1] + sequence_length_range[0])

            if dropped_frames > 0:
                print("Last {} frames not used for trajectory {}".format(dropped_frames,
                                                                         trajectory_name))

            sequence_lengths = [sequence_length_range[0]] * len(start_frames)
            sequence = {"sequence_length": sequence_lengths, "start_frame_index": start_frames}
            dataframe = pd.DataFrame(sequence)

            if "trajectory_segments" in serialization_destination:
                del serialization_destination["trajectory_segments"]

            serialization_destination.create_dataset('trajectory_segments', data=dataframe.to_numpy())

        elif len(sequence_length_range) == 2:
            sequence = {"sequence_length": [], "start_frame_index": []}
            start = 0
            while True:
                sequence_length = np.random.randint(sequence_length_range[0], sequence_length_range[1] + 1)
                if start + sequence_length < trajectory_length:
                    sequence["sequence_length"].append(sequence_length)
                    sequence["start_frame_index"].append(start)
                    start += (sequence_length - overlap)
                else:
                    if trajectory_length - start >= sequence_length_range[0]:
                        sequence["sequence_length"].append(trajectory_length - start)
                        sequence["start_frame_index"].append(start)
                    else:
                        print("Last {} frames not used for trajectory {}".format(trajectory_length - start,
                                                                                 trajectory_name))
                    break
            dataframe = pd.DataFrame(sequence)

            if "trajectory_segments" in serialization_destination:
                del serialization_destination["trajectory_segments"]

            serialization_destination.create_dataset('trajectory_segments', data=dataframe.to_numpy())
        else:
            raise InvalidSizeException("sequence_length_range should have length of either 1 or 2")


class MidAirDataSegmenter(TrajectorySegmenter):
    def __init__(self, path_to_data: str):
        self.path_to_data = path_to_data

    def segment(self, sequence_length_range: Tuple = None, overlap: int = 0):
        """Segments the trajectories found into, sequences of length sequence_length_range.
        If no arguments are passed or if sequence_length_range is None (default) then
        the whole trajectory will be used a a single segment.
        """

        for root, dirs, files in os.walk(self.path_to_data):
            if "sensor_records.hdf5" not in files:
                continue
            else:
                sensor_records = h5py.File(root + "/sensor_records.hdf5", "r+")
                for trajectory in sensor_records:
                    trajectory_length = self._get_trajectory_length(sensor_records[trajectory])
                    self.segment_trajectory(overlap, sensor_records[trajectory], sequence_length_range, trajectory,
                                            trajectory_length)

    def _get_trajectory_length(self, trajectory):
        for dataset in trajectory["camera_data"].values():
            return dataset.len()


class HDF5Opener:
    def __init__(self, root: str):
        """
        Factory that encapsulates the H5PY file instantiation process. To facilitate access to the HDF5
        file when used in a dataloader with multiple workers. This is required because H5PY does not
        support multi process reading and writing on a single H5PY File object.
        :param root:
        """
        self.root: str = root
        self.__HDF5__: h5py.File = None

    def __enter__(self) -> h5py.File:
        self.__HDF5__ = h5py.File(self.root + "/sensor_records.hdf5", "r")
        assert self.__HDF5__ is not None
        return self.__HDF5__.__enter__()

    def __exit__(self, *args):
        assert self.__HDF5__ is not None
        self.__HDF5__.__exit__(args)


class Segment(AbstractSegment):
    def __init__(self, root: str, trajectory: str, camera_view: str, start_frame_index: int, segment_length: int,
                 hdf5: HDF5Opener):

        super().__init__(segment_length)
        self.camera_view: str = camera_view
        self.hdf5: HDF5Opener = hdf5
        self.start_frame_index: int = start_frame_index
        self.trajectory: str = trajectory
        self.root: str = root

    def get_positions(self) -> torch.Tensor:
        """Return position as tensor"""
        position = super(Segment, self).get_positions()
        position = self._rotate_world_frame_to_camera_frame(position)

        return position

    def _rotate_world_frame_to_camera_frame(self, position: torch.Tensor) -> torch.Tensor:
        return torch.mm(position, torch.Tensor(self._get_to_camera_frame_rotation_matrix()))

    def _get_raw_positions(self) -> torch.Tensor:
        position = []
        for i in range(self.start_frame_index, self.start_frame_index + self.segment_length):
            with self.hdf5 as hdf5:
                position.append(hdf5[self.trajectory]["groundtruth"]["position"][i])
        return torch.Tensor(position)

    def get_images(self) -> list:
        image_sequence = []

        for path in self.get_images_path():
            image_sequence.append(Image.open(path))

        return image_sequence

    def get_images_path(self) -> list:
        paths = []

        for index in range(self.start_frame_index, self.start_frame_index + self.segment_length):
            with self.hdf5 as hdf5:
                camera_view = hdf5[self.trajectory]["camera_data"][self.camera_view]
                paths.append(self.root + "/" + camera_view[index])

        return paths

    def _reset_attitude_origin(self, attitude: list) -> list:
        """Reset the sequence's rotation relative to the first frame. Expects angles to be in quaternions"""
        initial_orientation_quat = attitude[0]
        resetted_attitude = []

        for i, orientation in enumerate(attitude):
            current_orientation_quat = orientation
            resetted_orientation_quat = initial_orientation_quat.inverse * current_orientation_quat
            resetted_orientation_quat = self._rotate_quaternion_to_camera_frame(resetted_orientation_quat)

            if i == 0:
                numpy.testing.assert_almost_equal(resetted_orientation_quat.elements,
                                                  numpy.asarray([1.0, 0.0, 0.0, 0.0]))

            resetted_attitude.append(resetted_orientation_quat)

        return resetted_attitude

    def _rotate_quaternion_to_camera_frame(self, resetted_oriention_quat):
        # Rotate the i,j,k (x,y,z) component of the quaternion to the camera frame using the rotation matrix
        camera_frame_quat = numpy.append(resetted_oriention_quat.elements[:1],
                                         resetted_oriention_quat.elements[1:].dot(
                                             self._get_to_camera_frame_rotation_matrix()))
        return Quaternion(camera_frame_quat)

    def _get_raw_attitude_as_quat(self) -> List[Quaternion]:
        return self._get_raw_attitude()

    def _get_raw_attitude(self) -> list:
        """
        Return list of attitude as a Quaternion object
        """
        attitude = []
        for i in range(self.start_frame_index, self.start_frame_index + self.segment_length):
            # MidAir uses scalar first notation to represent quaternions just like PyQuaternion so no re-ordering
            # of the components of the quaternions required. Note that if using Scipy.Rotation. You will
            # have to use the scalar last format.
            with self.hdf5 as hdf5:
                attitude.append(Quaternion(hdf5[self.trajectory]["groundtruth"]["attitude"][i]).unit)
        return attitude

    def _get_to_camera_frame_rotation_matrix(self) -> numpy.ndarray:
        """
        Rotation matrix describing the tranformation from the world frame of midair (x: forward, y:right, z:down)
        to the camera frame (x: right, y:down, z:forward)
        @return:
        """
        to_camera_frame_rotation_matrix = numpy.asarray([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]]).T
        return to_camera_frame_rotation_matrix


class MidAirImageSequenceDataset(AbstractSegmentDataset):
    def data_is_relative(self) -> bool:
        return False

    def __init__(self, dataset_path: str, resize_mode='crop', new_size=None, img_mean: float = None,
                 img_std: float = None, minus_point_5: bool = False, trajectories: list = None, augment_dataset: bool = False):
        super().__init__(dataset_path, framerate=25,
                         resize_mode=resize_mode,
                         new_size=new_size,
                         img_mean=img_mean,
                         img_std=img_std,
                         minus_point_5=minus_point_5,
                         augment_dataset=augment_dataset)
        self.trajectories: list = trajectories
        self.HDF5: dict = self._load_hdf5_file_openers()
        self.segments: list = self._load_segments()

    def _load_segments(self):
        segments = []
        for root, hdf5 in self.HDF5.items():
            segments.extend(SegmentMapper.map_all(root, hdf5, trajectories=self.trajectories))

        return segments

    def _load_hdf5_file_openers(self) -> dict:
        HDF5 = {}
        for root, dirs, files in os.walk(self.dataset_path):
            if "sensor_records.hdf5" not in files:
                continue
            else:
                HDF5[root] = self._get_sensor_record_opener(root)
        assert len(HDF5) > 0, "No dataset file (hdf5) found."
        return HDF5

    def _get_sensor_record_opener(self, root: str) -> HDF5Opener:
        return HDF5Opener(root)


class MidAirImageSequenceDatasetEulerDifferences(MidAirImageSequenceDataset):

    def __init__(self, dataset_path: str, resize_mode='crop', new_size=None, img_mean: float = None,
                 img_std: float = None,
                 minus_point_5: bool = False, trajectories: list = None, augment_dataset: bool = False):
        super().__init__(dataset_path, resize_mode, new_size, img_mean, img_std, minus_point_5, trajectories, augment_dataset)

    def __getitem__(self, item: int):
        segment, image_sequence = super().__getitem__(item)
        return image_sequence, self._get_segment_pose(segment)

    def get_absolute_pose_for_item(self, item: int):
        segment, _ = super().__getitem__(item)
        position = segment.get_positions()
        attitude = segment.get_attitudes()
        # output is intrinsic Tait-Bryan angles following the y-x'-z''
        attitude = torch.Tensor(numpy.around(Geometry.quaternions_to_tait_bryan_rotations(attitude), 7))

        numpy.testing.assert_almost_equal(attitude[0], numpy.asarray([0.0, 0.0, 0.0]))

        return torch.cat([attitude, position], 1)

    def _get_segment_pose(self, segment: Segment) -> torch.Tensor:
        position = segment.get_position_differences()
        attitude = segment.get_attitude_differences()
        # output is intrinsic Tait-Bryan angles following the y-x'-z''
        attitude = torch.Tensor(numpy.around(Geometry.quaternions_to_tait_bryan_rotations(attitude), 7))

        return torch.cat([attitude, position], 1)

    def data_is_relative(self) -> bool:
        return True

class SegmentMapper:

    @classmethod
    def map_all(cls, root: str, hdf5_opener: HDF5Opener, trajectories: list = None):
        segments = []
        with hdf5_opener as hdf5:
            for trajectory_name, trajectory in hdf5.items():

                if trajectories is not None and trajectory_name not in trajectories:
                    continue

                segment_dataframe = cls._load_segments_dataframe(trajectory)
                for index, raw_segment in segment_dataframe.iterrows():
                    for camera_view in trajectory["camera_data"].keys():
                        segments.append(cls._map(root, trajectory_name, camera_view, raw_segment, hdf5_opener))
        return segments

    @classmethod
    def _load_segments_dataframe(cls, trajectory):
        return pd.DataFrame(trajectory["trajectory_segments"],
                            columns=["sequence_length", "start_frame_index"])

    @classmethod
    def _map(cls, root: str, trajectory: str, camera_view: str, raw_segment: dict, hdf5: HDF5Opener):
        return Segment(root, trajectory, camera_view, raw_segment["start_frame_index"], raw_segment["sequence_length"],
                       hdf5)


class MidAirIteratorGenerator:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def get_image_path_iterator(self):
        for root, dirs, files in os.walk(self.dataset_path):
            if os.path.exists(root + "/sensor_records.hdf5"):
                sensor_records = h5py.File(root + "/sensor_records.hdf5", "r+")
                for trajectory in sensor_records:
                    for camera_view in sensor_records[trajectory]["camera_data"]:
                        for frame in sensor_records[trajectory]["camera_data"][camera_view]:
                            yield camera_view, root + "/" + frame


class MidAirDataPreprocessor(AbstractDataPreprocessor):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.dataset_path = dataset_path
        self.iteratorGenerator = MidAirIteratorGenerator(self.dataset_path)

    def clean(self):
        for root, dirs, files in os.walk(self.dataset_path):
            self._clean_sensor_records(dirs, files, root)
            self._clean_frames(files, root)

    def _get_image_path_iterator(self):
        for _, path in self.iteratorGenerator.get_image_path_iterator():
            yield path

    def _reduce_records_to_framerate(self, groundtruth_group):
        for groundtruth_sensor_record in groundtruth_group:
            # only keep every 4th element in the groundtruth_sensor_record (will reduce the reading from 100Hz to 25Hz or 25fps)
            tmp = groundtruth_group[groundtruth_sensor_record][0::4]
            del groundtruth_group[groundtruth_sensor_record]
            groundtruth_group[groundtruth_sensor_record] = tmp

    def _clean_frames(self, files, root):
        if "frames.zip" in files:
            with zipfile.ZipFile(root + "/frames.zip", "r") as zip_ref:
                zip_ref.extractall(root)
                os.remove(root + "/frames.zip")

    def _clean_sensor_records(self, dirs, files, root):
        if "sensor_records.zip" in files:
            with zipfile.ZipFile(root + "/sensor_records.zip", "r") as zip_ref:
                zip_ref.extractall(root)

        if os.path.exists(root + "/sensor_records.hdf5"):
            sensor_records = h5py.File(root + "/sensor_records.hdf5", "r+")

            for trajectory in sensor_records:
                try:
                    del sensor_records[trajectory]["gps"]
                except KeyError:
                    print("No GPS record in trajectory {} in {}".format(trajectory, root))
                try:
                    del sensor_records[trajectory]["imu"]
                except KeyError:
                    print("No IMU record in trajectory {} in {}".format(trajectory, root))
                try:
                    del sensor_records[trajectory]["groundtruth"]["acceleration"]
                except KeyError:
                    print("No acceleration record in trajectory {} in {}".format(trajectory, root))
                try:
                    del sensor_records[trajectory]["groundtruth"]["angular_velocity"]
                except KeyError:
                    print("No angular_velocity record in trajectory {} in {}".format(trajectory, root))
                try:
                    del sensor_records[trajectory]["groundtruth"]["velocity"]
                except KeyError:
                    print("No velocity record in trajectory {} in {}".format(trajectory, root))

                self._reduce_records_to_framerate(sensor_records[trajectory]["groundtruth"])

                for camera_view in sensor_records[trajectory]["camera_data"]:
                    if camera_view not in dirs:
                        try:
                            del sensor_records[trajectory]["camera_data"][camera_view]
                        except KeyError:
                            print("No view {} found in trajectory {} in {}".format(camera_view, trajectory, root))
                    else:
                        for ground_truth_group in sensor_records[trajectory]["groundtruth"]:
                            assert len(sensor_records[trajectory]["camera_data"][camera_view]) == len(sensor_records[trajectory]["groundtruth"][ground_truth_group])
