import math
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy
import torch
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader

from Common.Helpers import Geometry, cuda_is_available
from Common.Trajectory import Trajectory, TrajectoryPlotter
from Datasets.Interfaces import AbstractSegmentDataset
from Logging import CometLogger
from Metrics.Metrics import MetricLogger, TranslationRotationDrift, AbsoluteTrajectoryError, RelativeError, \
    CompoundTranslationRotationDrift, CompoundAbsoluteTrajectoryError, CompoundRelativeError, TrajectoryStats
from Models.Losses import RMSELoss


class ModelTester:
    def __init__(self, model: nn.Module, trajectory_dataloaders: List[Tuple[str, str, DataLoader]],
                 sliding_window_size: int, sliding_window_overlap: int,
                 model_name: str):
        self.model: nn.Module = model
        self.trajectory_dataloaders: List[Tuple[str, str, DataLoader]] = trajectory_dataloaders
        self.sliding_window_size = sliding_window_size
        assert sliding_window_overlap >= 1, "sliding_window_overlap should be greater or equal to 1"
        self.sliding_window_overlap = sliding_window_overlap

        # Number of new frames between sliding windows
        self.step = self.sliding_window_size - self.sliding_window_overlap
        self.loss = RMSELoss()
        self.model_name = model_name

    def _trim_trajectories(self, location_gt):
        max_lenght = 1000
        location_differences_gt = location_gt[1:] - location_gt[0:-1]

        last_included_index = 0
        total_lenght = 0
        for vector in location_differences_gt:
            lenght = total_lenght + math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
            if lenght < max_lenght:
                total_lenght = lenght
                last_included_index = last_included_index + 1
            else:
                break

        return last_included_index + 1



    def run(self):
        trajectory_rotation_losses = []
        trajectory_translation_losses = []
        drift_errors = []
        ATEs = []
        REs = []

        for dataset_name, trajectory_name, dataloader in self.trajectory_dataloaders:
            dataset: AbstractSegmentDataset = dataloader.dataset
            print("testing {}, {}".format(trajectory_name, dataset_name))

            start = time.time()
            predictions, rotation_losses, translation_losses, absolute_ground_truth = self._test(dataloader)
            end = time.time()

            last_included_index = self._trim_trajectories(absolute_ground_truth[:, 3:])
            predictions = predictions[:last_included_index+1]
            absolute_ground_truth = absolute_ground_truth[:last_included_index+1]


            CometLogger.print(f"Inferred {len(predictions)} poses in {end-start} seconds.\n"
                              f"Dataset fps: {dataset.framerate}, inference fps {len(predictions)/(end-start)}.")

            trajectory_rotation_losses.append((dataset_name, trajectory_name, rotation_losses))
            trajectory_translation_losses.append((dataset_name, trajectory_name, translation_losses))

            plotter = TrajectoryPlotter(trajectory_name, dataset_name, self.model_name, absolute_ground_truth, predictions)
            CometLogger.get_experiment().log_figure(figure=plotter.rotation_figure,
                                                    figure_name='rotation {} {}'.format(trajectory_name, dataset_name))

            CometLogger.get_experiment().log_figure(figure=plotter.position_figure,
                                                    figure_name='translation {} {}'.format(trajectory_name,
                                                                                           dataset_name))

            drift, ATE, RE = self._log_metrics(absolute_ground_truth, dataset, dataset_name, predictions, trajectory_name)
            drift_errors.append(drift)
            ATEs.append(ATE)
            REs.append(RE)

            self._log_matrix_poses(predictions, absolute_ground_truth, dataset_name, trajectory_name)
            self._log_quaternion_poses(predictions, absolute_ground_truth, dataset_name, trajectory_name)

        self._log_compounded_metrics(ATEs, REs, drift_errors)

        losses_figure = self._plot_trajectory_losses(trajectory_rotation_losses, trajectory_translation_losses)
        CometLogger.get_experiment().log_figure(figure=losses_figure, figure_name="trajectory_losses")

        # compute total avg losses
        translation_loss = self._complute_total_avg_loss(trajectory_translation_losses)
        rotation_loss = self._complute_total_avg_loss(trajectory_rotation_losses)

        CometLogger.get_experiment().log_metric("Total Avg Translation loss (test phase)", translation_loss)
        CometLogger.get_experiment().log_metric("Total Avg Rotation loss (test phase)", rotation_loss)

    def _test(self, trajectory_dataloader: DataLoader):
        """
        Performs an inference pass on the trajectory.
        @param trajectory_dataloader:
        @return:
        """
        self.model.eval()

        rotation_losses = []
        translation_losses = []
        predictions = Trajectory(trajectory_dataloader.dataset.data_is_relative(),
                                 is_groundtruth=False,
                                 sliding_window_size=self.sliding_window_size,
                                 sliding_window_overlap=self.sliding_window_overlap)
        ground_truth = Trajectory(trajectory_dataloader.dataset.data_is_relative(),
                                 is_groundtruth=True,
                                 sliding_window_size=self.sliding_window_size,
                                 sliding_window_overlap=self.sliding_window_overlap)

        for segments_batch in trajectory_dataloader:
            x, batch_target = segments_batch

            if cuda_is_available():
                x = x.cuda()
                batch_target = torch.squeeze(batch_target.cuda())

            prediction = self.model.forward(x)
            if type(prediction) is tuple:
                batch_predict = torch.squeeze(prediction[0])
            else:
                batch_predict = torch.squeeze(prediction)

            batch_target = torch.squeeze(batch_target)
            rotation_loss = self.loss.compute(batch_predict[:, :3], batch_target[:, :3])
            translation_loss = self.loss.compute(batch_predict[:, 3:], batch_target[:, 3:])
            rotation_losses.append(float(rotation_loss.data.cpu().numpy()))
            translation_losses.append(float(translation_loss.data.cpu().numpy()))

            batch_predict = batch_predict.detach().cpu().numpy()
            batch_target = batch_target.cpu().numpy()

            for batch_element_id in range(0, batch_target.shape[0]):

                # If there's only 1 element in the batch, process the whole batch, otherwise, process element wise
                if len(batch_target.shape) == 2:
                    predict = batch_predict
                    target = batch_target
                else:
                    predict = numpy.squeeze(batch_predict[batch_element_id])
                    target = numpy.squeeze(batch_target[batch_element_id])

                ground_truth.append(target)
                predictions.append(predict)

                if len(batch_target.shape) == 2:
                    break

        return predictions.assembled_pose, rotation_losses, translation_losses, ground_truth.assembled_pose

    def _log_quaternion_poses(self, poses, poses_gt, dataset_name: str, trajectory_name: str):
        """
        Logs the pose in text format where the angle is a quaternion:
            timestamp tx ty tz qx qy qz qw
        """
        pose_output = ""
        pose_gt_output = ""

        for i, pose in enumerate(poses):
            # att.elements[[1, 2, 3, 0]] reorganizes quaternion elements
            # from scalar first w-x-y-z to scalar last x-y-z-w
            rotation_quat = Geometry.tait_bryan_rotation_to_quaternion(pose[:3]).elements[[1, 2, 3, 0]]
            rotation_quat_gt = Geometry.tait_bryan_rotation_to_quaternion(poses_gt[i][:3]).elements[[1, 2, 3, 0]]

            pose_output = pose_output + f"{i} {pose[3]} {pose[4]} {pose[5]} " \
                                        f"{rotation_quat[0]} {rotation_quat[1]} {rotation_quat[2]} {rotation_quat[3]}"
            pose_gt_output = pose_gt_output + f"{i} {poses_gt[i][3]} {poses_gt[i][4]} {poses_gt[i][5]} " \
                                        f"{rotation_quat_gt[0]} {rotation_quat_gt[1]} {rotation_quat_gt[2]} " \
                                        f"{rotation_quat_gt[3]}"
            if i < len(poses) - 1:
                pose_output = pose_output + "\n"
                pose_gt_output = pose_gt_output + "\n"

        metadata = dict()
        metadata["title"] = "pose_output_quaternion"
        metadata["dataset"] = dataset_name
        metadata["trajectory"] = trajectory_name
        metadata["model"] = self.model_name
        filename = f'{metadata["title"]}_{dataset_name}_{trajectory_name}_{metadata["model"]}.txt'
        CometLogger.get_experiment().log_asset_data(pose_output, name=filename, metadata=metadata)

        metadata["title"] = "pose_gt_output_quaternion"
        filename = f'{metadata["title"]}_{dataset_name}_{trajectory_name}_{metadata["model"]}.txt'
        CometLogger.get_experiment().log_asset_data(pose_gt_output, name=filename, metadata=metadata)

    def _log_matrix_poses(self, poses, poses_gt, dataset_name: str, trajectory_name: str):
        """
        Logs the pose in text format where the angle is a rotation matrix:
            T00 T01 T02 T03
            T10 T11 T12 T13
            T20 T21 T22 T23
            0   0   0   1

            T00 T01 T02 T03 T10 T11 T12 T13 T20 T21 T22 T23
        """
        pose_output = ""
        pose_gt_output = ""
        matrices = Geometry.poses_to_transformations_matrix(poses[:, 3:], poses[:, :3])
        matrices_gt = Geometry.poses_to_transformations_matrix(poses_gt[:, 3:], poses_gt[:, :3])

        for i, _ in enumerate(matrices):
            pose_matrix = matrices[i]
            pose_matrix_gt = matrices_gt[i]

            pose_output = pose_output + f"{pose_matrix[0][0]} {pose_matrix[0][1]} {pose_matrix[0][2]} {pose_matrix[0][3]} " \
                                        f"{pose_matrix[1][0]} {pose_matrix[1][1]} {pose_matrix[1][2]} {pose_matrix[1][3]} " \
                                        f"{pose_matrix[2][0]} {pose_matrix[2][1]} {pose_matrix[2][2]} {pose_matrix[2][3]}"

            pose_gt_output = pose_gt_output + f"{pose_matrix_gt[0][0]} {pose_matrix_gt[0][1]} {pose_matrix_gt[0][2]} {pose_matrix_gt[0][3]} " \
                                        f"{pose_matrix_gt[1][0]} {pose_matrix_gt[1][1]} {pose_matrix_gt[1][2]} {pose_matrix_gt[1][3]} " \
                                        f"{pose_matrix_gt[2][0]} {pose_matrix_gt[2][1]} {pose_matrix_gt[2][2]} {pose_matrix_gt[2][3]}"
            if i < len(poses) - 1:
                pose_output = pose_output + "\n"
                pose_gt_output = pose_gt_output + "\n"

        metadata = dict()
        metadata["title"] = "pose_output_matrix"
        metadata["dataset"] = dataset_name
        metadata["trajectory"] = trajectory_name
        metadata["model"] = self.model_name
        filename = f'{metadata["title"]}_{dataset_name}_{trajectory_name}_{metadata["model"]}.txt'
        CometLogger.get_experiment().log_asset_data(pose_output, name=filename, metadata=metadata)

        metadata["title"] = "pose_gt_output_matrix"
        filename = f'{metadata["title"]}_{dataset_name}_{trajectory_name}_{metadata["model"]}.txt'
        CometLogger.get_experiment().log_asset_data(pose_gt_output, name=filename, metadata=metadata)

    def _log_compounded_metrics(self, ATEs, REs, drift_errors):
        metric_logger = MetricLogger()

        compound_drift_errors = CompoundTranslationRotationDrift(self.model_name, drift_errors)
        metric_logger.log(compound_drift_errors)
        CometLogger.get_experiment().log_metric("avg_translation_error_percent",
                                                compound_drift_errors.metrics["avg_translation_error_percent"])
        CometLogger.get_experiment().log_metric("avg_rotation_error_degrees_per_meter",
                                                compound_drift_errors.metrics["avg_rotation_error_degrees_per_meter"])

        compound_ATE = CompoundAbsoluteTrajectoryError(self.model_name, ATEs)
        metric_logger.log(compound_ATE)
        CometLogger.get_experiment().log_metric("ATE_trans_RMSE",
                                                compound_ATE.metrics["absolute_trajectory_error"]
                                                ['ATE_trans_stats']["rmse"])
        CometLogger.get_experiment().log_metric("ATE_rot_degrees_RMSE",
                                                compound_ATE.metrics["absolute_trajectory_error"]
                                                ['ATE_rot_stats']["rmse"])

        compound_RE = CompoundRelativeError(self.model_name, REs)
        metric_logger.log(compound_RE)

    def _log_metrics(self, absolute_ground_truth, dataset, dataset_name, predictions, trajectory_name):
        metric_logger = MetricLogger()
        stats = TrajectoryStats(dataset_name,
                                trajectory_name,
                                self.model_name,
                                absolute_ground_truth[:, 3:],
                                dataset.framerate)
        metric_logger.log(stats)
        drift = TranslationRotationDrift(dataset_name,
                                         trajectory_name,
                                         self.model_name,
                                         absolute_ground_truth[:, 3:],
                                         absolute_ground_truth[:, :3],
                                         predictions[:, 3:],
                                         predictions[:, :3],
                                         dataset.framerate)
        metric_logger.log(drift)
        ATE = AbsoluteTrajectoryError(dataset_name,
                                      trajectory_name,
                                      self.model_name,
                                      absolute_ground_truth[:, 3:],
                                      absolute_ground_truth[:, :3],
                                      predictions[:, 3:],
                                      predictions[:, :3])
        metric_logger.log(ATE)
        RE = RelativeError(dataset_name,
                           trajectory_name,
                           self.model_name,
                           absolute_ground_truth[:, 3:],
                           absolute_ground_truth[:, :3],
                           predictions[:, 3:],
                           predictions[:, :3])
        metric_logger.log(RE)
        return drift, ATE, RE

    def _complute_total_avg_loss(self, trajectory_losses: list) -> float:
        nbr_of_losses = 0
        loss = 0
        for trajectory_losses in trajectory_losses:
            loss += sum(trajectory_losses[2])
            nbr_of_losses += len(trajectory_losses[2])
        return loss / nbr_of_losses

    def _plot_trajectory_losses(self, trajectory_rotation_losses: list, trajectory_translation_losses: list) -> Figure:
        fig, axs = plt.subplots(2, figsize=(8, 11))
        fig.suptitle("Avg RMSE vs Trajectories")
        axs[0].set_title("Avg Rotations RMS Error vs trajectory")
        axs[0].bar(["{}, {}".format(trajectory[1], trajectory[0]) for trajectory in trajectory_rotation_losses],
                   [sum(trajectory[2]) / len(trajectory[2]) for trajectory in trajectory_rotation_losses])
        axs[0].set(xlabel='trajectories', ylabel='Avg Rotation RMSE')

        axs[1].set_title("Avg Translation RMS Error vs trajectory")
        axs[1].bar(["{}, {}".format(trajectory[1], trajectory[0]) for trajectory in trajectory_translation_losses],
                   [sum(trajectory[2]) / len(trajectory[2]) for trajectory in trajectory_translation_losses])
        axs[1].set(xlabel='trajectories', ylabel='Avg translation RMSE')

        return fig
