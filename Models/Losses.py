from abc import ABC, abstractmethod

import numpy
import torch
from torch import nn

from Common.Helpers import TensorGeometry


class AbstractLoss(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def compute(self, prediction: torch.Tensor, target: torch.Tensor):
        pass


class RMSELoss(AbstractLoss):
    def __init__(self, eps=1e-7):
        """
        Computes the RMSE loss. Inspired by: https://discuss.pytorch.org/t/rmse-loss-function/16540/4
        @param eps: small value added to prevent NaNs when backpropagating the sqrt of 0
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def compute(self, prediction, target: torch.Tensor):
        loss = torch.sqrt(self.mse(prediction, target) + self.eps)
        return loss


class BatchSegmentMSELoss(AbstractLoss):
    def __init__(self):
        super().__init__()

    def compute(self, prediction: torch.Tensor, target: torch.Tensor):
        # predicted and y dimensions are as follows: (batch, seq, dim_pose)
        # Weighted MSE Loss
        assert prediction.shape == target.shape
        angle_loss = torch.nn.functional.mse_loss(prediction[:, :, :3], target[:, :, :-3])
        translation_loss = torch.nn.functional.mse_loss(prediction[:, :, 3:], target[:, :, -3:])
        loss = (100 * angle_loss + translation_loss)
        return loss


class LoopedBatchSegmentMSELoss(BatchSegmentMSELoss):
    def __init__(self, loop_weight=1, MSELoss_weight=1, loop_radius_threshold=6):
        super().__init__()
        self.MSELoss_weight = MSELoss_weight
        self.loop_weight = loop_weight
        self.loop_loss = LoopLoss(loop_radius_threshold)

    def compute(self, prediction: torch.Tensor, target: torch.Tensor):
        lloss = self.loop_loss.compute(prediction, target)
        MSELoss = super(LoopedBatchSegmentMSELoss, self).compute(prediction, target)

        return self.MSELoss_weight * MSELoss + self.loop_weight * lloss


class EulerToMatrixGlobalMSELoss(AbstractLoss):
    def __init__(self):
        super().__init__()

    def compute(self, prediction: torch.Tensor, target: torch.Tensor):
        assert prediction.shape == target.shape and target.shape[-1] == 6

        return self._compute_absolute_angle_loss(prediction, target) \
               + self._compute_absolute_translation_loss(prediction, target)

    def _compute_absolute_angle_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        prediction_absolute_rot_matrices = TensorGeometry.batch_assembleDeltaRotationMatrices(
            TensorGeometry.batchEulerAnglesToRotationMatrixTensor(prediction[:, :, :-3]))
        target_absolute_rot_matrices = TensorGeometry.batch_assembleDeltaRotationMatrices(
            TensorGeometry.batchEulerAnglesToRotationMatrixTensor(target[:, :, :-3]))

        return 100 * torch.nn.functional.mse_loss(prediction_absolute_rot_matrices, target_absolute_rot_matrices)

    def _compute_absolute_translation_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        predicted_absolute_translation = TensorGeometry.batch_assembleDeltaTranslationMatrices(
            prediction[:, :, -3:])
        target_absolute_translation = TensorGeometry.batch_assembleDeltaTranslationMatrices(
            target[:, :, -3:])

        return torch.nn.functional.mse_loss(predicted_absolute_translation, target_absolute_translation)


class EulerGlobalMSELoss(AbstractLoss):
    def __init__(self):
        super().__init__()

    def compute(self, prediction: torch.Tensor, target: torch.Tensor):
        assert prediction.shape == target.shape and target.shape[-1] == 6

        return self._compute_absolute_angle_loss(prediction, target) \
               + self._compute_absolute_translation_loss(prediction, target)

    def _compute_absolute_angle_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        prediction = TensorGeometry.batch_assembleDeltaEulerAngles(prediction[:, :, :-3])
        target = TensorGeometry.batch_assembleDeltaEulerAngles(target[:, :, :-3])

        return 100 * torch.nn.functional.mse_loss(prediction, target)

    def _compute_absolute_translation_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        prediction = TensorGeometry.batch_assembleDeltaTranslationMatrices(prediction[:, :, -3:])
        target = TensorGeometry.batch_assembleDeltaTranslationMatrices(target[:, :, -3:])

        return torch.nn.functional.mse_loss(prediction, target)


class EulerGlobalRelativeMSELoss(AbstractLoss):
    def __init__(self, relative_loss_weight=1, absolute_loss_weight=1):
        super().__init__()
        self.absolute_loss_weight = absolute_loss_weight
        self.relative_loss_weight = relative_loss_weight

    def compute(self, prediction: torch.Tensor, target: torch.Tensor):
        assert prediction.shape == target.shape and target.shape[-1] == 6

        return self.relative_loss_weight * self._compute_relative_loss(prediction, target) + \
               self.absolute_loss_weight * self._compute_absolute_loss(prediction, target)

    def _compute_relative_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        loss = BatchSegmentMSELoss()
        return loss.compute(prediction, target)

    def _compute_absolute_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        loss = EulerGlobalMSELoss()
        return loss.compute(prediction, target)


class LoopedEulerGlobalRelativeMSELoss(EulerGlobalRelativeMSELoss):
    def __init__(self, relative_loss_weight=1, absolute_loss_weight=1, loop_loss_weight=1, loop_radius_threshold=6):
        super().__init__(relative_loss_weight, absolute_loss_weight)
        self.loop_loss_weight = loop_loss_weight
        self.loop_loss = LoopLoss(loop_radius_threshold)

    def compute(self, prediction: torch.Tensor, target: torch.Tensor):
        assert prediction.shape == target.shape and target.shape[-1] == 6
        glob_rel_loss = super(LoopedEulerGlobalRelativeMSELoss, self).compute(prediction, target)
        lloss = self.loop_loss.compute(prediction, target)

        return glob_rel_loss + self.loop_loss_weight * lloss


class EulerSeparateGlobalRelativeMSELoss(AbstractLoss):
    def __init__(self, global_loss_weight=1, relative_loss_weight=1):
        super().__init__()
        self.relative_loss_weight = relative_loss_weight
        self.global_loss_weight = global_loss_weight

    def compute(self, prediction: tuple, target: torch.Tensor):
        assert prediction[0].shape == target.shape and target.shape[-1] == 6

        return self.global_loss_weight * self._compute_global_loss(prediction[1], target) + \
               self.relative_loss_weight * self._compute_relative_loss(prediction[0], target)

    def _compute_relative_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        loss = BatchSegmentMSELoss()
        return loss.compute(prediction, target)

    def _compute_global_loss(self, prediction: torch.Tensor, target: torch.Tensor):
        global_rotations = TensorGeometry.batch_assembleDeltaEulerAngles(target[:, :, :-3])
        global_translations = TensorGeometry.batch_assembleDeltaTranslationMatrices(target[:, :, -3:])
        global_pose_minus_start = torch.cat((global_rotations, global_translations), dim=2)[:, 1:, :]

        loss = BatchSegmentMSELoss()
        return loss.compute(prediction, global_pose_minus_start)


class TemporalGeometricConsistencyLoss(AbstractLoss):
    def __init__(self, tempgeo_loss_weight=1, batchMSE_loss_weight=1):
        super().__init__()
        self.batchMSE_loss_weight = batchMSE_loss_weight
        self.tempgeo_loss_weight = tempgeo_loss_weight
        self.batchSegmentMSELoss = BatchSegmentMSELoss()

    def compute(self, prediction: torch.Tensor, target: torch.Tensor):
        assert prediction.shape == target.shape and target.shape[-1] == 6
        groundtruth_mse_loss = self.batchSegmentMSELoss.compute(prediction, target)
        relative_consistency_loss = self._compute_temporal_geometric_consistency(prediction, target)
        return self.tempgeo_loss_weight * relative_consistency_loss + self.batchMSE_loss_weight * groundtruth_mse_loss

    def _compute_temporal_geometric_consistency(self, predictions: torch.Tensor, target: torch.Tensor):
        rotation_loss = self._compute_rotation_temporal_geometric_consistency(predictions[:, :, :3],
                                                                              target[:, :, :3])
        translation_loss = self._compute_translation_temporal_geometric_consistency(predictions[:, :, 3:],
                                                                                    target[:, :, 3:])

        return 100 * rotation_loss + translation_loss

    def _compute_rotation_temporal_geometric_consistency(self, predictions: torch.Tensor,
                                                         target: torch.Tensor):
        return self._rigid_body_tranformation_composition_loss(target,
                                                               predictions,
                                                               TensorGeometry.batch_assembleDeltaEulerAngles)

    def _compute_translation_temporal_geometric_consistency(self, predictions: torch.Tensor,
                                                            target: torch.Tensor):
        return self._rigid_body_tranformation_composition_loss(target,
                                                               predictions,
                                                               TensorGeometry.batch_assembleDeltaTranslationMatrices)

    def _rigid_body_tranformation_composition_loss(self, target,
                                                   prediction,
                                                   composition_function):
        device = target.device
        diff1 = target
        diff2 = prediction

        # Level 1 loss (ex: Loss frame 0-1, L12, L23, etc.)
        loss = torch.nn.functional.mse_loss(diff2, diff1)

        # sublevel losses (ex: Loss frame 0-2, L24, L04, etc.)
        n = 0
        while 2 ** n + 1 <= diff1.shape[1]:
            j = 2 ** n
            nbr_new_elements = 0
            assemble1 = torch.empty(diff1.shape, requires_grad=diff1.requires_grad).to(device)
            assemble2 = torch.empty(diff2.shape, requires_grad=diff2.requires_grad).to(device)
            for i in range(diff1.shape[1] - 1):
                if (i + j) + 1 > diff1.shape[1]:
                    break
                nbr_new_elements = i + 1
                assemble1[:, i, :] = composition_function(diff1[:, i:i + j + 1:j, :])[:, -1, :]
                assemble2[:, i, :] = composition_function(diff2[:, i:i + j + 1:j, :])[:, -1, :]

            diff1 = assemble1[:, :nbr_new_elements, :]
            diff2 = assemble2[:, :nbr_new_elements, :]
            loss += torch.nn.functional.mse_loss(diff2, diff1)
            n += 1

            # Ensure that the computation was properly done
            if not (2 ** n + 1 <= diff1.shape[1]):
                numpy.testing.assert_almost_equal(diff1[:, 0, :].data.cpu().numpy(),
                                                  composition_function(target[:, :2 ** n + 1, :])[:, 2 ** n, :]
                                                  .data.cpu().numpy(),
                                                  decimal=6)

        return loss


class LoopedTemporalGeometricConsistencyLoss(TemporalGeometricConsistencyLoss):
    def __init__(self, tempgeo_loss_weight=1, batchMSE_loss_weight=1, loop_weight=1, loop_radius_threshold=6):
        super().__init__(tempgeo_loss_weight, batchMSE_loss_weight)
        self.loop_radius_threshold = loop_radius_threshold
        self.loop_weight = loop_weight
        self.loop_loss = LoopLoss(loop_radius_threshold)

    def compute(self, prediction: torch.Tensor, target: torch.Tensor):
        assert prediction.shape == target.shape and target.shape[-1] == 6
        temp_geo_loss = super(LoopedTemporalGeometricConsistencyLoss, self).compute(prediction, target)
        lloss = self.loop_loss.compute(prediction, target)

        return temp_geo_loss + self.loop_weight * lloss


class GlobalRelativeTemporalGeometricConsistencyLoss(TemporalGeometricConsistencyLoss):
    def __init__(self, tempgeo_loss_weight=1, batchMSE_loss_weight=1):
        super().__init__(tempgeo_loss_weight, batchMSE_loss_weight)
        self.eulerSeparateGlobalRelativeMSELoss = EulerSeparateGlobalRelativeMSELoss()

    def compute(self, prediction: tuple, target: torch.Tensor):
        assert prediction[0].shape == target.shape and target.shape[-1] == 6

        groundtruth_mse_loss = self.eulerSeparateGlobalRelativeMSELoss.compute(prediction, target)
        relative_consistency_loss = self._compute_temporal_geometric_consistency(prediction[0], target)

        return self.tempgeo_loss_weight * relative_consistency_loss + self.batchMSE_loss_weight * groundtruth_mse_loss


class LoopLoss(AbstractLoss):

    def __init__(self, loop_radius_threshold=6):
        """
        @param radius_threshold: Default value is 6 meters as suggested in
        Zhang et Al. “Graph-Based Place Recognition in Image Sequences with CNN Features”
        """
        super().__init__()
        self.loop_radius_threshold = loop_radius_threshold

    def compute(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Minimises the distance between locations that are within the radius defined by self.radius_threshold.
        The lowest value to which the distance can be minimized is the actual groundtruth distance between
        the two points.
        @param prediction: [rotations, translations] matrix where translations are in meters
        @param target: [rotations, translations] matrix where translations are in meters
        """
        predicted_positions = TensorGeometry.batch_assembleDeltaTranslationMatrices(prediction[:, :, -3:])
        target_positions = TensorGeometry.batch_assembleDeltaTranslationMatrices(target[:, :, -3:])
        loops = self._compute_loop_matrix(target_positions)
        loops = loops.type(torch.FloatTensor).to(prediction.device)

        predicted_distances = self._location_distances(predicted_positions)
        target_distances = self._location_distances(target_positions)

        # Use the loops' positions as boolean indices to select only the loops in the predicted and target
        # distances matrix
        predicted_loops_distances = loops * predicted_distances
        target_loops_distances = loops * target_distances

        return torch.nn.functional.mse_loss(predicted_loops_distances, target_loops_distances)

    def _compute_loop_matrix(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Based on the distances between each positions relative to each other, identifies the ones that lie within
        the radius threshold, those will be considered loops.
        @param positions:
        @return:
        """
        distances = self._location_distances(positions)
        loops = distances <= self.loop_radius_threshold
        return loops

    def _location_distances(self, positions) -> torch.Tensor:
        """
        Compute euclidean distance between each positions relative to each other
        @param positions:
        @return:
        """
        diff = positions[..., None, :, :] - positions[..., None, :]
        distances = torch.norm(diff, dim=3)
        return distances


class ATELoss(AbstractLoss):
    def __init__(self):
        super().__init__()

    def compute(self, prediction: torch.Tensor, target: torch.Tensor):
        assert prediction.shape == target.shape

        translation_loss = self._compute_translation_ATE(prediction[:, :, 3:], target[:, :, -3:])
        angle_loss = self._compute_rotational_ATE(prediction[:, :, :3], target[:, :, :-3])

        loss = (100 * angle_loss + translation_loss)
        return loss

    def _compute_translation_ATE(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Modified from: https://github.com/uzh-rpg/rpg_trajectory_evaluation
        @param prediction:
        @param target:
        @return:
        """
        e_trans_vec = (target - prediction)
        e_trans = torch.sqrt(torch.sum(e_trans_vec ** 2, 2))

        return torch.mean(e_trans)

    def _compute_rotational_ATE(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Ins[ired by: https://github.com/uzh-rpg/rpg_trajectory_evaluation
        @param prediction:
        @param target:
        @return:
        """

        target_rot_matrix = TensorGeometry.batchEulerAnglesToRotationMatrixTensor(target)
        prediction_rot_matrix = TensorGeometry.batchEulerAnglesToRotationMatrixTensor(prediction)
        e_rot = torch.empty(target_rot_matrix.shape, requires_grad=prediction.requires_grad).to(prediction.device)

        for i, segment in enumerate(prediction_rot_matrix):
            for j, rotation in enumerate(segment):
                e_rot[i, j] = torch.mm(target_rot_matrix[i, j], rotation.inverse())

        e_rot = TensorGeometry.batchRotationMatrixTensorToEulerAngles(e_rot)

        return torch.mean(e_rot)
