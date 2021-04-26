import json
import math
from abc import ABC, abstractmethod
from typing import List

import numpy
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from Common.Helpers import Geometry
from Logging import CometLogger
from .KITTI_odometry_metrics.kitti_odometry import KittiEvalOdom
from .rpg_trajectory_evaluation.results_writer import compute_statistics
from .rpg_trajectory_evaluation.trajectory import Trajectory


class Metric(ABC):
    def __init__(self, dataset_name: str, trajectory_name: str, model_name: str):
        self._init_instance_variables(dataset_name, model_name, trajectory_name)

    def _init_instance_variables(self, dataset_name, model_name, trajectory_name):
        self.trajectory_name = trajectory_name
        self.model_name = model_name
        self.metrics: dict = dict()
        self._set_dataset_name(dataset_name)
        self.metrics["trajectory"] = trajectory_name
        self.metrics["model"] = model_name
        self.figures_loader: Metric._FigureLoader = Metric._FigureLoader(self._load_figures)

    def _set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name
        self.metrics["dataset"] = dataset_name

    @abstractmethod
    def _load_figures(self) -> dict:
        pass

    class _FigureLoader:
        """
        This private class is a resource manager for figure object since objects can be quite heavy.
        It will compute the figure when they are needed and it will automatically delete
        the figures once you're done using them.

        Usage:
            with metric.figures as figures:
                #do something

        """

        def __init__(self, load_figures_in_dict_function):
            self.load_figures_in_dict_function = load_figures_in_dict_function

        def __enter__(self) -> dict:
            self._figures: dict = self.load_figures_in_dict_function()
            return self._figures

        def __exit__(self, exc_type, exc_val, exc_tb):
            del self._figures


class TrajectoryStats(Metric):

    def __init__(self, dataset_name: str,
                 trajectory_name: str,
                 model_name: str,
                 location_gt: numpy.ndarray,
                 framerate: int):
        super().__init__(dataset_name, trajectory_name, model_name)
        self.metrics["framerate"] = framerate
        self.metrics["trajectory_lenght"], self.vector_lenghts = self._calculate_trajectory_lenght(location_gt)
        self.metrics["avg_trajectory_speed"] = self._calculate_avg_trajectory_speed(self.metrics["trajectory_lenght"],
                                                                                    len(location_gt),
                                                                                    self.metrics["framerate"])
        self.metrics["x_displacement_range"] = self._calculate_displacement_range(0, location_gt)
        self.metrics["y_displacement_range"] = self._calculate_displacement_range(1, location_gt)
        self.metrics["z_displacement_range"] = self._calculate_displacement_range(2, location_gt)

    def _calculate_trajectory_lenght(self, location_gt: numpy.ndarray):
        location_differences_gt = location_gt[1:] - location_gt[0:-1]
        vector_lenghts = [math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) for vector in location_differences_gt]
        return numpy.sum(vector_lenghts), vector_lenghts

    def _calculate_avg_trajectory_speed(self, trajectory_lenght: int, nbr_of_frames: int, framerate: int):
        total_time = nbr_of_frames / framerate
        return trajectory_lenght/total_time

    def _calculate_displacement_range(self, axis: int, location_gt):
        min = numpy.min(location_gt[:, axis])
        max = numpy.max(location_gt[:, axis])

        return min, max


    def _load_figures(self) -> dict:
        return dict()


class TranslationRotationDrift(Metric):
    """
    Generate translation/rotation error that measures drift from the ground-truth as is done in
    the KITTI development kit. (See: http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
    """

    def __init__(self, dataset_name: str,
                 trajectory_name: str,
                 model_name: str,
                 location_gt: numpy.ndarray,
                 orientation_gt: numpy.ndarray,
                 location: numpy.ndarray,
                 orientation: numpy.ndarray, framerate: int,
                 stepsize: int = 10, align: bool = False):
        """
        @param framerate: in frames per seconds
        @param stepsize: will evaluate at evey X frames
        """
        super().__init__(dataset_name, trajectory_name, model_name)

        self.metrics["framerate"] = framerate
        self.metrics["stepsize"] = stepsize

        pose_gt = Geometry.poses_to_transformations_matrix(location_gt, orientation_gt)
        pose = Geometry.poses_to_transformations_matrix(location, orientation)
        self.kittiEvalOdom = KittiEvalOdom(framerate, stepsize)

        if align:
            pose = self.kittiEvalOdom.align_poses("7dof", pose_gt, pose)

        self.seq_errors = self.kittiEvalOdom.calc_sequence_errors(pose_gt, pose)
        self.metrics["segment_errors"]: dict = TranslationRotationDrift.map_segment_error_details_to_segment_error(
            self.seq_errors)

        self.metrics["average_segment_errors"]: dict = self.kittiEvalOdom.compute_segment_error_from_sequence_Error(
            self.seq_errors)
        avg_translation_error, avg_rotation_error = self.kittiEvalOdom.compute_overall_error_from_sequence_error(
            self.seq_errors)
        self.metrics["avg_translation_error"] = avg_translation_error
        self.metrics["avg_rotation_error_radians"] = avg_rotation_error
        self.metrics["avg_translation_error_percent"] = avg_translation_error * 100
        self.metrics["avg_rotation_error_degrees_per_meter"] = avg_rotation_error / numpy.pi * 180 * 100

    def _load_figures(self) -> dict:
        figures = dict()
        figures["average_segment_error_plot"] = self.kittiEvalOdom.plot_average_segment_errors(
            self.metrics["average_segment_errors"],
            self.trajectory_name,
            self.dataset_name,
            self.model_name)
        return figures

    @staticmethod
    def map_segment_error_details_to_segment_error(seq_errors) -> dict:
        segment_errors = dict()
        for seq in seq_errors:
            segment_details = dict()
            segment_details["start_frame"] = seq[0]
            segment_details["rotation_error_percent"] = seq[1]
            segment_details["translation_error_percent"] = seq[2]
            segment_details["segment_lenght"] = seq[3]
            segment_details["speed"] = seq[4]

            if seq[3] in segment_errors.keys():
                segment_errors[seq[3]].append(segment_details)
            else:
                segment_errors[seq[3]] = []
                segment_errors[seq[3]].append(segment_details)
        return segment_errors


class CompoundTranslationRotationDrift(TranslationRotationDrift):
    def __init__(self,
                 model_name: str,
                 trans_rot_drifts: List[TranslationRotationDrift],
                 stepsize: int = 10):
        self._init_instance_variables(None, model_name, "all")
        self.trans_rot_drifts = trans_rot_drifts
        self.metrics["stepsize"] = stepsize
        dataset_names = set()
        all_seq_errors = []
        avg_translation_errors = []
        avg_rotation_error_radians = []
        avg_translation_error_percent = []
        avg_rotation_error_degrees_per_meter = []

        for error in trans_rot_drifts:
            dataset_names.add(error.dataset_name)
            all_seq_errors.extend(error.seq_errors)

            avg_translation_errors.append(error.metrics["avg_translation_error"])
            avg_rotation_error_radians.append(error.metrics["avg_rotation_error_radians"])
            avg_translation_error_percent.append(error.metrics["avg_translation_error_percent"])
            avg_rotation_error_degrees_per_meter.append(error.metrics["avg_rotation_error_degrees_per_meter"])

        self.metrics["avg_avg_translation_error"] = numpy.sum(avg_translation_errors) / len(avg_translation_errors)
        self.metrics["avg_avg_rotation_error_radians"] = numpy.sum(avg_rotation_error_radians) / len(avg_rotation_error_radians)
        self.metrics["avg_avg_translation_error_percent"] = numpy.sum(avg_translation_error_percent) / len(avg_translation_error_percent)
        self.metrics["avg_avg_rotation_error_degrees_per_meter"] = numpy.sum(avg_rotation_error_degrees_per_meter) / len(avg_rotation_error_degrees_per_meter)

        self.kittiEvalOdom = KittiEvalOdom(0, stepsize)
        self.metrics["segment_errors"]: dict = TranslationRotationDrift.map_segment_error_details_to_segment_error(
            all_seq_errors)
        self.metrics["average_segment_errors"]: dict = self.kittiEvalOdom.compute_segment_error_from_sequence_Error(
            all_seq_errors)
        avg_translation_error, avg_rotation_error = self.kittiEvalOdom.compute_overall_error_from_sequence_error(
            all_seq_errors)
        self.metrics["avg_translation_error"] = avg_translation_error
        self.metrics["avg_rotation_error_radians"] = avg_rotation_error
        self.metrics["avg_translation_error_percent"] = avg_translation_error * 100
        self.metrics["avg_rotation_error_degrees_per_meter"] = avg_rotation_error / numpy.pi * 180 * 100

        self._set_dataset_name(str(dataset_names)[1:-1])


class RPGMetric(Metric):
    """
    Absolute Trajectory Error and Relative Errors as computed in:
    Zichao Zhang, Davide Scaramuzza: A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry,
    IEEE/RSJ Int. Conf. Intell. Robot. Syst. (IROS), 2018.

    Using code modified from https://github.com/uzh-rpg/rpg_trajectory_evaluation
    """

    def _map_location_orientation_to_rpg_trajectory(self, location, location_gt, orientation, orientation_gt):
        # att.elements[[1, 2, 3, 0]] reorganizes quaternion elements from scalar first w-x-y-z to scalar last x-y-z-w
        orientation_gt = [quat.elements[[1, 2, 3, 0]] for quat in
                          Geometry.tait_bryan_rotations_to_quaternions(orientation_gt)]
        orientation = [quat.elements[[1, 2, 3, 0]] for quat in
                       Geometry.tait_bryan_rotations_to_quaternions(orientation)]
        orientation_gt = numpy.array(orientation_gt)
        orientation = numpy.array(orientation)
        return Trajectory(location, location_gt, orientation, orientation_gt)


class AbsoluteTrajectoryError(RPGMetric):
    """
    Absolute Trajectory Error as computed in:
    Zichao Zhang, Davide Scaramuzza: A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry,
    IEEE/RSJ Int. Conf. Intell. Robot. Syst. (IROS), 2018.

    Using code modified from https://github.com/uzh-rpg/rpg_trajectory_evaluation
    """

    def __init__(self, dataset_name: str, trajectory_name: str, model_name: str, location_gt: numpy.ndarray,
                 orientation_gt: numpy.ndarray, location: numpy.ndarray, orientation: numpy.ndarray):
        super().__init__(dataset_name, trajectory_name, model_name)
        trajectory = self._map_location_orientation_to_rpg_trajectory(location, location_gt, orientation,
                                                                      orientation_gt)
        self.metrics["absolute_trajectory_error"] = trajectory.compute_absolute_error()

    def _load_figures(self) -> dict:
        return dict()


class CompoundAbsoluteTrajectoryError(AbsoluteTrajectoryError):
    def __init__(self, model_name: str,
                 ATE_errors: List[AbsoluteTrajectoryError]):

        self._init_instance_variables(None, model_name, "all")
        self.metrics["absolute_trajectory_error"] = dict()
        self.metrics["absolute_trajectory_error"]['ATE_trans_L2_norm'] = []
        self.metrics["absolute_trajectory_error"]['ATE_rot_degrees'] = []
        self._ATE_errors = ATE_errors
        dataset_names = set()

        avg_avg_ATE_trans_L2_norm = []
        avg_avg_ATE_rot_degrees = []

        for error in self._ATE_errors:
            dataset_names.add(error.dataset_name)
            self.metrics["absolute_trajectory_error"]['ATE_trans_L2_norm'] \
                .extend(error.metrics["absolute_trajectory_error"]['ATE_trans_L2_norm'])
            self.metrics["absolute_trajectory_error"]['ATE_rot_degrees'] \
                .extend(error.metrics["absolute_trajectory_error"]['ATE_rot_degrees'])

            avg_avg_ATE_trans_L2_norm.append(error.metrics["absolute_trajectory_error"]["ATE_trans_stats"]["mean"])
            avg_avg_ATE_rot_degrees.append(error.metrics["absolute_trajectory_error"]["ATE_rot_stats"]["mean"])

        self.metrics["avg_avg_ATE_trans_L2_norm"] = numpy.sum(avg_avg_ATE_trans_L2_norm) / len(avg_avg_ATE_trans_L2_norm)
        self.metrics["avg_avg_ATE_rot_degrees"] = numpy.sum(avg_avg_ATE_rot_degrees) / len(avg_avg_ATE_rot_degrees)

        self.metrics["absolute_trajectory_error"]['ATE_trans_stats'] = compute_statistics(
            self.metrics["absolute_trajectory_error"]['ATE_trans_L2_norm'])

        self.metrics["absolute_trajectory_error"]['ATE_rot_stats'] = compute_statistics(
            self.metrics["absolute_trajectory_error"]['ATE_rot_degrees'])

        self._set_dataset_name(str(dataset_names)[1:-1])

    def _load_figures(self) -> dict:
        figures = dict()
        figures["ATE_translation_box_plot"] = self._compute_l2_norm_boxplot(self._ATE_errors)
        figures["ATE_rotation_box_plot"] = self._compute_rot_degrees_boxplot(self._ATE_errors)
        return figures

    def _compute_l2_norm_boxplot(self, ATE_errors: List[AbsoluteTrajectoryError]):

        data = []
        ticks = []

        for error in ATE_errors:
            data.append(error.metrics["absolute_trajectory_error"]['ATE_trans_L2_norm'])
            ticks.append(error.trajectory_name)

        fig, axs = plt.subplots(2)
        fig.set_size_inches(15, 15)
        fig.suptitle(
            "ATE translation L2 norm ({}, {}, {})".format(self.trajectory_name, self.dataset_name, self.model_name))
        axs[0].set_title("L2 norms per Trajectories")
        axs[0].boxplot(data)
        axs[0].set(xlabel='Trajectories', ylabel='L2 Norm (m)')
        axs[0].set_xticklabels(ticks)

        axs[1].set_title("L2 norms for all trajectories")
        axs[1].boxplot(self.metrics["absolute_trajectory_error"]['ATE_trans_L2_norm'])
        axs[1].set(xlabel='All trajectories', ylabel='L2 Norm (m)')

        return fig

    def _compute_rot_degrees_boxplot(self, ATE_errors: List[AbsoluteTrajectoryError]):

        data = []
        ticks = []

        for error in ATE_errors:
            data.append(error.metrics["absolute_trajectory_error"]['ATE_rot_degrees'])
            ticks.append(error.trajectory_name)

        fig, axs = plt.subplots(2)
        fig.set_size_inches(15, 15)
        fig.suptitle(
            "ATE Rotation (deg) ({}, {}, {})".format(self.trajectory_name, self.dataset_name, self.model_name))
        axs[0].set_title("Rotation Error (deg) per Trajectories")
        axs[0].boxplot(data)
        axs[0].set(xlabel='Trajectories', ylabel='Rotation Error (deg)')
        axs[0].set_xticklabels(ticks)

        axs[1].set_title("Rotation Error (deg) for all trajectories")
        axs[1].boxplot(self.metrics["absolute_trajectory_error"]['ATE_rot_degrees'])
        axs[1].set(xlabel='All trajectories', ylabel='Rotation Error (deg)')

        return fig


class RelativeError(RPGMetric):
    """
    Relative Error as computed in:
    Zichao Zhang, Davide Scaramuzza: A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry,
    IEEE/RSJ Int. Conf. Intell. Robot. Syst. (IROS), 2018.

    Using code modified from https://github.com/uzh-rpg/rpg_trajectory_evaluation
    """

    def __init__(self, dataset_name: str, trajectory_name: str, model_name: str, location_gt: numpy.ndarray,
                 orientation_gt: numpy.ndarray, location: numpy.ndarray, orientation: numpy.ndarray):
        super().__init__(dataset_name, trajectory_name, model_name)
        trajectory = self._map_location_orientation_to_rpg_trajectory(location, location_gt, orientation,
                                                                      orientation_gt)
        self.metrics["relative_error"] = trajectory.compute_relative_errors()

    def _load_figures(self) -> dict:
        figures = dict()
        figures["RE_boxplot"] = self._compute_per_lenght_RE_boxplot(self.metrics)
        return figures

    def _compute_per_lenght_RE_boxplot(self, metrics: dict):

        data = []
        ticks = []

        for lenght in metrics["relative_error"].keys():
            data.append(metrics["relative_error"][lenght]["rel_trans"])
            ticks.append(lenght)

        fig, axs = plt.subplots(4)
        fig.set_size_inches(8, 25)
        fig.suptitle(
            "Relative Errors ({}, {}, {})".format(self.trajectory_name, self.dataset_name, self.model_name))
        axs[0].set_title("Translation Relative Error (m) per Distance Traveled")
        axs[0].boxplot(data)
        axs[0].set(xlabel='Distance Traveled (m)', ylabel='Relative Error - Translation (m)')
        axs[0].set_xticklabels(ticks)

        data = []

        for lenght in metrics["relative_error"].keys():
            data.append(metrics["relative_error"][lenght]["rel_trans_perc"])

        axs[1].set_title("Translation Relative Error (%) per Distance Traveled")
        axs[1].boxplot(data)
        axs[1].set(xlabel='Distance Traveled (m)', ylabel='Relative Error - Translation (%)')
        axs[1].set_xticklabels(ticks)

        data = []

        for lenght in metrics["relative_error"].keys():
            data.append(metrics["relative_error"][lenght]["rel_trans"])

        axs[2].set_title("Rotation Relative Error (deg) per Distance Traveled")
        axs[2].boxplot(data)
        axs[2].set(xlabel='Distance Traveled (m)', ylabel='Relative Error - Rotation (deg)')
        axs[2].set_xticklabels(ticks)

        data = []

        for lenght in metrics["relative_error"].keys():
            data.append(metrics["relative_error"][lenght]["rel_rot_deg_per_m"])

        axs[3].set_title("Rotation Relative Error (deg/m) per Distance Traveled")
        axs[3].boxplot(data)
        axs[3].set(xlabel='Distance Traveled (m)', ylabel='Relative Error - Rotation (deg/m)')
        axs[3].set_xticklabels(ticks)

        return fig


class CompoundRelativeError(RelativeError):
    def __init__(self, model_name: str, RE_errors: List[RelativeError]):

        self._init_instance_variables(None, model_name, "all")
        self.metrics["relative_error"] = dict()
        dataset_names = set()

        for error in RE_errors:
            dataset_names.add(error.dataset_name)
            for i in error.metrics["relative_error"]:
                if i not in self.metrics["relative_error"].keys():
                    self.metrics["relative_error"][i] = dict()
                    self.metrics["relative_error"][i]["rel_trans"] = []
                    self.metrics["relative_error"][i]["rel_trans_perc"] = []
                    self.metrics["relative_error"][i]["rel_rot"] = []
                    self.metrics["relative_error"][i]["rel_yaw"] = []
                    self.metrics["relative_error"][i]["rel_gravity"] = []
                    self.metrics["relative_error"][i]["rel_rot_deg_per_m"] = []

                self.metrics["relative_error"][i]["rel_trans"].extend(error.metrics["relative_error"][i]["rel_trans"])
                self.metrics["relative_error"][i]["rel_trans_perc"].extend(
                    error.metrics["relative_error"][i]["rel_trans_perc"])
                self.metrics["relative_error"][i]["rel_rot"].extend(error.metrics["relative_error"][i]["rel_rot"])
                self.metrics["relative_error"][i]["rel_yaw"].extend(error.metrics["relative_error"][i]["rel_yaw"])
                self.metrics["relative_error"][i]["rel_gravity"].extend(
                    error.metrics["relative_error"][i]["rel_gravity"])
                self.metrics["relative_error"][i]["rel_rot_deg_per_m"].extend(
                    error.metrics["relative_error"][i]["rel_rot_deg_per_m"])

        for i in self.metrics["relative_error"]:
            self.metrics["relative_error"][i]["rel_trans_stats"] = compute_statistics(
                self.metrics["relative_error"][i]["rel_trans"])
            self.metrics["relative_error"][i]["rel_trans_perc_stats"] = compute_statistics(
                self.metrics["relative_error"][i]["rel_trans_perc"])
            self.metrics["relative_error"][i]["rel_rot_stats"] = compute_statistics(
                self.metrics["relative_error"][i]["rel_rot"])
            self.metrics["relative_error"][i]["rel_yaw_stats"] = compute_statistics(
                self.metrics["relative_error"][i]["rel_yaw"])
            self.metrics["relative_error"][i]["rel_gravity_stats"] = compute_statistics(
                self.metrics["relative_error"][i]["rel_gravity"])
            self.metrics["relative_error"][i]["rel_rot_deg_per_m_stats"] = compute_statistics(
                self.metrics["relative_error"][i]["rel_rot_deg_per_m"])

        self._set_dataset_name(str(dataset_names)[1:-1])


class MetricLogger:

    def _log_text_asset(self, text: str, title: str, dataset_name: str, trajectory_name: str, model_name: str):
        metadata = dict()
        metadata["title"] = title
        metadata["dataset"] = dataset_name
        metadata["trajectory"] = trajectory_name
        metadata["model"] = model_name
        filename = f"{title}_{dataset_name}_{trajectory_name}_{model_name}.json"

        CometLogger.get_experiment().log_asset_data(text, name=filename, metadata=metadata)

    def _log_figure(self, fig: Figure, figure_name: str):
        CometLogger.get_experiment().log_figure(figure=fig, figure_name=figure_name)

    def log(self, metric: Metric):
        metrics = self._arrays_to_lists(metric.metrics)
        self._log_text_asset(json.dumps(metrics),
                             type(metric).__name__,
                             metric.dataset_name,
                             metric.trajectory_name,
                             metric.model_name)
        with metric.figures_loader as figures:
            for fig_name in figures.keys():
                name_to_log = fig_name + "_{}_{}".format(metric.trajectory_name, metric.dataset_name)
                self._log_figure(figures[fig_name], name_to_log)

    def _arrays_to_lists(self, metrics: dict) -> dict:
        metrics_copy = dict(metrics)
        for key in metrics.keys():
            if isinstance(metrics[key], dict):
                metrics_copy[key] = self._arrays_to_lists(metrics[key])
            elif isinstance(metrics[key], numpy.ndarray):
                metrics_copy[key] = metrics[key].tolist()
            else:
                continue
        return metrics_copy
