from typing import List
from unittest import TestCase

import numpy
import torch
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

from Common.Helpers import Geometry
from Common.Trajectory import Trajectory
from Datasets.Interfaces import AbstractSegment
from ModelTester import ModelTester


class TestModelTester(TestCase):

    def setUp(self):
        # yaw rotation of 1 degree and displacement of 1 meter on the x axis
        self.absolute_pose_deg = numpy.array([[0, 0, 0, 0, 0, 0],
                                              [1, 0, 0, 1, 0, 0],
                                              [2, 0, 0, 2, 0, 0],
                                              [3, 0, 0, 3, 0, 0],
                                              [4, 0, 0, 4, 0, 0]])
        # Degrees to radians
        self.absolute_pose = numpy.concatenate(
                (Rotation.from_euler("YXZ", self.absolute_pose_deg[:, :3], degrees=True).as_euler("YXZ"),
                 self.absolute_pose_deg[:, 3:]), axis=1)
        self.segment1 = TestModelTester.MockEulerSegment(3, self.absolute_pose[:3])
        self.segment2 = TestModelTester.MockEulerSegment(3, self.absolute_pose[-3:])
        self.relative_poses_segment_1 = self._get_segment_pose(self.segment1)
        self.relative_poses_segment_2 = self._get_segment_pose(self.segment2)
        self.mock_dataloader = TestModelTester.MockRelativeDataloader()

    def test_given_relative_target_poses_should_return_valid_absolute_poses(self):

        trajectory = Trajectory(is_input_data_relative=True, is_groundtruth=True, sliding_window_size=3,
                                sliding_window_overlap=1)

        trajectory.append(self.relative_poses_segment_1)
        trajectory.append(self.relative_poses_segment_2)

        numpy.testing.assert_almost_equal(trajectory.assembled_pose, self.absolute_pose)

    def test_given_relative_predicted_poses_should_return_valid_absolute_poses(self):
        trajectory = Trajectory(is_input_data_relative=True, is_groundtruth=True, sliding_window_size=3,
                                sliding_window_overlap=1)

        trajectory.append(self.relative_poses_segment_1)
        trajectory.append(self.relative_poses_segment_2)

        numpy.testing.assert_almost_equal(trajectory.assembled_pose, self.absolute_pose)

    def _get_segment_pose(self, segment: AbstractSegment) -> numpy.ndarray:
        position = segment.get_position_differences()
        attitude = segment.get_attitude_differences()
        # att.elements[[1, 2, 3, 0]] reorganizes quaternion elements from scalar first w-x-y-z to scalar last x-y-z-w
        # output is intrinsic Tait-Bryan angles following the y-x'-z''
        attitude = numpy.array([numpy.around(Rotation.from_quat(att.elements[[1, 2, 3, 0]]).as_euler("YXZ"), 7)
                                      for att in attitude])

        return numpy.concatenate((attitude, position), 1)

    class MockRelativeDataset:
        def data_is_relative(self) -> bool:
            return True

    class MockRelativeDataloader:
        def __init__(self):
            self.dataset = TestModelTester.MockRelativeDataset()

    class MockEulerSegment(AbstractSegment):

        def __init__(self, segment_length: int, pose: numpy.ndarray):
            super().__init__(segment_length)
            #Convert pose to radians
            self.pose: numpy.ndarray = pose

        def get_images(self) -> list:
            pass

        def get_images_path(self) -> list:
            pass

        def _get_raw_attitude(self):
            return self.pose[:, :3]

        def _reset_attitude_origin(self, raw_attitude: List[Quaternion]):
            return Geometry.reset_orientations_to_origin(raw_attitude[0], raw_attitude)

        def _get_raw_positions(self) -> torch.Tensor:
            return torch.Tensor(self.pose[:, 3:])

        def _get_raw_attitude_as_quat(self) -> List[Quaternion]:
            return Geometry.tait_bryan_rotations_to_quaternions(self._get_raw_attitude())