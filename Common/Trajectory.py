from typing import Iterable, List

import numpy
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from Common.Helpers import Geometry


class Trajectory:
    def __init__(self,
                 is_input_data_relative: bool,
                 is_groundtruth: bool,
                 sliding_window_size: int,
                 sliding_window_overlap: int):
        self._sliding_window_overlap = sliding_window_overlap
        self._sliding_window_size = sliding_window_size
        self._is_groundtruth = is_groundtruth
        self._is_input_data_relative = is_input_data_relative
        self.assembled_pose: list = None
        self._step = self._sliding_window_size - self._sliding_window_overlap

    def append(self, pose_fragment: numpy.ndarray):
        if self._is_groundtruth:
            assembled_fragment = self._relative_target_pose_to_absolute_pose(self.assembled_pose, pose_fragment)
        else:
            assembled_fragment = self._relative_predicted_pose_to_absolute_pose(self.assembled_pose, pose_fragment)

        if self.assembled_pose is None:
            self.assembled_pose = assembled_fragment
        else:
            self.assembled_pose = numpy.append(self.assembled_pose, numpy.asarray(assembled_fragment[-self._step:]),
                                               axis=0)

    def _relative_predicted_pose_to_absolute_pose(self, previous_iteration_pose, pose):
        """
        Transforms relative pose data to absolute pose
        @param previous_iteration_pose:
        @param pose:
        @return:
        """
        if previous_iteration_pose is None and self._is_input_data_relative:
            translations = Geometry.assemble_delta_translations(pose[:, 3:])
            rotations = Geometry.assemble_delta_tait_bryan_rotations(pose[:, :3])
            return numpy.concatenate((rotations, translations), axis=1)
        elif previous_iteration_pose is not None and not self._is_input_data_relative:
            translations = Geometry.get_position_differences(pose[:, 3:])
            rotations = Geometry.get_tait_bryan_orientation_differences(pose[:, :3])
        else:
            translations = pose[:, 3:]
            rotations = pose[:, :3]

        # only keep the non overlapping predictions
        relative_pose_overlap = self._sliding_window_overlap - 1
        translations = translations[relative_pose_overlap:]
        rotations = rotations[relative_pose_overlap:]

        translations = Geometry.remap_position_axes(
            Geometry.tait_bryan_rotation_to_quaternion(previous_iteration_pose[-1, :3]),
            translations)

        # assemble with the previous iteration poses
        temp_translation = numpy.concatenate((previous_iteration_pose[-1, 3:].reshape((1, 3)), translations), axis=0)
        positions = Geometry.assemble_delta_translations(temp_translation)[2:]

        temp_rotations = numpy.concatenate((previous_iteration_pose[-1, :3].reshape((1, 3)), rotations), axis=0)
        orientations = Geometry.assemble_delta_tait_bryan_rotations(temp_rotations)[2:]
        orientations = numpy.around(orientations, 7)
        return numpy.concatenate((orientations, positions), axis=1)

    def _relative_target_pose_to_absolute_pose(self, previous_iteration_pose, pose):
        """
        Transforms relative pose data to absolute pose
        @param previous_iteration_pose:
        @param pose:
        @return:
        """
        if self._is_input_data_relative:
            translations = Geometry.assemble_delta_translations(pose[:, 3:])
            rotations = Geometry.assemble_delta_tait_bryan_rotations(pose[:, :3])
        else:
            translations = pose[:, 3:]
            rotations = pose[:, :3]

        if previous_iteration_pose is None:
            translations = Geometry.remap_position_axes(
                Geometry.tait_bryan_rotation_to_quaternion(rotations[0]), translations)
            translations = Geometry.reset_positions_to_origin(translations[0], translations)
            rotations = Geometry.reset_euler_orientations_to_origin(rotations[0], rotations)
        else:
            translations = Geometry.remap_position_axes(
                Geometry.tait_bryan_rotation_to_quaternion(previous_iteration_pose[-self._sliding_window_overlap, :3]),
                translations)
            translations = Geometry.reset_positions_to_origin(
                previous_iteration_pose[-self._sliding_window_overlap, 3:],
                translations)
            rotations = Geometry.reset_euler_orientations_to_origin(
                previous_iteration_pose[-self._sliding_window_overlap, :3],
                rotations)
        rotations = numpy.around(rotations, 7)
        return numpy.concatenate((rotations, translations), axis=1)


class LiveTrajectory(Trajectory):
    def append(self, pose_fragment: numpy.ndarray):
        """Append new pose fragments to the live trajectory,"""
        if self._is_groundtruth:
            assembled_fragment = self._relative_target_pose_to_absolute_pose(self.assembled_pose, pose_fragment)
        else:
            assembled_fragment = self._relative_predicted_pose_to_absolute_pose(self.assembled_pose, pose_fragment)

        if self.assembled_pose is None:
            self.assembled_pose = assembled_fragment
        else:
            self.assembled_pose = numpy.append(self.assembled_pose, numpy.asarray(assembled_fragment[-self._step:]),
                                               axis=0)


class TrajectoryPlotter:
    def __init__(self,
                 trajectory_name: str,
                 dataset_name: str,
                 model_name: str,
                 groundtruth: Iterable = None,
                 predictions: Iterable = None
                 ):
        self.model_name = model_name
        self.position_figure, self._position_axs = plt.subplots(2, figsize=(6.4, 11))
        self.position_figure.suptitle("Positions ({}, {}, {})".format(trajectory_name, dataset_name, model_name))

        self._position_axs[0].set_title("X-Z plane (top view)")
        self._position_axs[0].set(xlabel='X', ylabel='Z')

        self._position_axs[1].set_title("X-Y plane (Side view) ({}, {})".format(trajectory_name, dataset_name))
        self._position_axs[1].set(xlabel='X', ylabel='Y')

        self.rotation_figure, self._rotation_axs = plt.subplots(3, figsize=(6.4, 15))
        self.rotation_figure.suptitle("Rotations ({}, {}, {})".format(trajectory_name, dataset_name, model_name))

        self._rotation_axs[0].set_title("X (Euler) Rotations vs Frame")
        self._rotation_axs[0].set(xlabel='Frames', ylabel='X')

        self._rotation_axs[1].set_title("Y (Euler) Rotations vs Frame")
        self._rotation_axs[1].set(xlabel='Frames', ylabel='Y')

        self._rotation_axs[2].set_title("Z (Euler) Rotations vs Frame")
        self._rotation_axs[2].set(xlabel='Frames', ylabel='Z')

        if groundtruth is not None or predictions is not None:
            self.update_position_plot(groundtruth, predictions)
            self.update_rotation_plot(groundtruth, predictions)

    def update_all_plots(self,
                         groundtruth: Iterable = None,
                         predictions: Iterable = None) -> List[Figure]:

        pos_fig = self.update_position_plot(groundtruth, predictions)
        rot_fig = self.update_rotation_plot(groundtruth, predictions)
        return [pos_fig, rot_fig]

    def update_position_plot(self,
                             groundtruth: Iterable = None,
                             predictions: Iterable = None) -> Figure:

        assert not (groundtruth is None and predictions is None), \
            "You must supply either a groundtruth or a prediction, or both"

        if groundtruth is not None:
            groundtruth = groundtruth[:, 3:]
            # plot X-Y plane
            self._position_axs[0].plot(groundtruth[:, 0], groundtruth[:, 2], label='Ground Truth', c="blue")
            # plot X-Z plane
            self._position_axs[1].plot(groundtruth[:, 0], groundtruth[:, 1], label='Ground Truth', c="blue")

        if predictions is not None:
            predictions = predictions[:, 3:]
            # plot X-Y plane
            self._position_axs[0].plot(predictions[:, 0], predictions[:, 2], c="red", label=self.model_name)
            # plot X-Z plane
            self._position_axs[1].plot(predictions[:, 0], predictions[:, 1], c="red", label=self.model_name)

        return self.position_figure

    def update_rotation_plot(self,
                             groundtruth: Iterable = None,
                             predictions: Iterable = None) -> Figure:

        assert not (groundtruth is None and predictions is None), \
            "You must supply either a groundtruth or a prediction, or both"

        if groundtruth is not None:
            groundtruth = groundtruth[:, :3]
            # plot X values
            self._rotation_axs[0].plot(range(0, len(groundtruth[:, 0])), groundtruth[:, 0], label='Ground Truth',
                                       c="blue")
            # plot Y values
            self._rotation_axs[1].plot(range(0, len(groundtruth[:, 1])), groundtruth[:, 1], label='Ground Truth',
                                       c="blue")
            # plot z values
            self._rotation_axs[2].plot(range(0, len(groundtruth[:, 2])), groundtruth[:, 2], label='Ground Truth',
                                       c="blue")

        if predictions is not None:
            predictions = predictions[:, :3]
            # plot X values
            self._rotation_axs[0].plot(range(0, len(predictions[:, 0])), predictions[:, 0], c="red",
                                       label=self.model_name)
            # plot Y values
            self._rotation_axs[1].plot(range(0, len(predictions[:, 1])), predictions[:, 1], c="red",
                                       label=self.model_name)
            # plot z values
            self._rotation_axs[2].plot(range(0, len(predictions[:, 2])), predictions[:, 2], c="red",
                                       label=self.model_name)

        return self.rotation_figure
