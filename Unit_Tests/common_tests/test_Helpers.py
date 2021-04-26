from unittest import TestCase

import numpy
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from Common.Helpers import TensorGeometry, Geometry
from Datasets.KITTI import Segment, KITTIImageSequenceDatasetEulerDifferences


class Test_TensorGeometry(TestCase):
    def test_euler_angles_to_rotation_matrix(self):
        # Intrinsic Tait-Bryan y-x'-z'' angles
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        input_tensor = torch.Tensor([y, x_prime, z_prime_prime]).requires_grad_(True)
        expected_matrix = Rotation.from_euler("YXZ", [y, x_prime, z_prime_prime]).as_matrix()
        actual_matrix = TensorGeometry.eulerAnglesToRotationMatrixTensor(input_tensor[0], input_tensor[1], input_tensor[2])
        self.assertTrue(actual_matrix.requires_grad)
        actual_matrix = actual_matrix.detach().numpy()
        numpy.testing.assert_allclose(expected_matrix, actual_matrix, rtol=3.22578393e-07)

    def test_batch_euler_to_matrix(self):
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        input_angles = [y, x_prime, z_prime_prime]

        input_tensor = torch.Tensor(
            [input_angles, input_angles, input_angles, input_angles, input_angles, input_angles, input_angles,
             input_angles]).requires_grad_(True)
        # 2 batches of 2 sequences of 2 frames
        input_tensor = input_tensor.view((2, 4, 3))

        actual_tensor = TensorGeometry.batchEulerAnglesToRotationMatrixTensor(input_tensor)
        self.assertTrue(actual_tensor.requires_grad)
        actual_tensor = actual_tensor.detach()
        flat_actual_tensor = torch.flatten(actual_tensor, start_dim=0, end_dim=1)

        expected_matrix = Rotation.from_euler("YXZ", [y, x_prime, z_prime_prime]).as_matrix()

        for actual_matrix in flat_actual_tensor:
            numpy.testing.assert_allclose(expected_matrix, actual_matrix, rtol=3.22578393e-07)

    def test_batch_matrix_to_euler(self):
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        input_angles = [y, x_prime, z_prime_prime]

        input_tensor = torch.Tensor(
            [input_angles, input_angles, input_angles, input_angles, input_angles, input_angles, input_angles,
             input_angles]).requires_grad_(True)
        # 2 batches of 2 sequences of 2 frames
        input_tensor = input_tensor.view((2, 4, 3))

        matrix_tensor = TensorGeometry.batchEulerAnglesToRotationMatrixTensor(input_tensor)
        self.assertTrue(matrix_tensor.requires_grad)
        euler_tensor = TensorGeometry.batchRotationMatrixTensorToEulerAngles(matrix_tensor)
        self.assertTrue(euler_tensor.requires_grad)
        flat_actual_tensor = torch.flatten(euler_tensor, start_dim=0, end_dim=1).detach()

        expected_matrix = input_tensor[1, 1, :].detach()

        for actual_matrix in flat_actual_tensor:
            numpy.testing.assert_allclose(expected_matrix, actual_matrix, rtol=3.22578393e-07)

    def test_rotation_matrix_to_euler(self):
        # Intrinsic Tait-Bryan x-y'-z'' angles
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        input_tensor = torch.Tensor(Rotation.from_euler("YXZ", [y, x_prime, z_prime_prime]).as_matrix()).requires_grad_(True)
        actual_matrix = TensorGeometry.rotation_matrix_to_euler_tait_bryan(input_tensor)
        self.assertTrue(actual_matrix.requires_grad)
        expected_matrix = numpy.asarray([y, x_prime, z_prime_prime])
        numpy.testing.assert_allclose(expected_matrix, actual_matrix.detach(), rtol=3.22578393e-07)

    def test_BatchRelativeRotationMatricesToAbsolute(self):
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        input_angles = [y, x_prime, z_prime_prime]

        relative_euler_rotation_batch = torch.Tensor(
            [input_angles, input_angles, input_angles, input_angles, input_angles, input_angles, input_angles,
             input_angles]).requires_grad_(True)

        # 2 segments of 4 euler rotations
        relative_euler_rotation_batch = relative_euler_rotation_batch.view((2, 4, 3))

        expected_relative_rotation_batch = TensorGeometry.batchEulerAnglesToRotationMatrixTensor(relative_euler_rotation_batch)
        self.assertTrue(expected_relative_rotation_batch.requires_grad)
        actual_absolute_orientation_batch = TensorGeometry.batch_assembleDeltaRotationMatrices(expected_relative_rotation_batch)
        self.assertTrue(actual_absolute_orientation_batch.requires_grad)

        actual_relative_rotation_batch = torch.zeros(expected_relative_rotation_batch.shape)

        #take the absolute rotation tensor and make the rotations relative
        for i, segment in enumerate(actual_absolute_orientation_batch):
            for j, rotation in enumerate(segment[1:]):
                actual_relative_rotation_batch[i, j] = torch.mm(rotation, segment[j].inverse())

        actual_relative_rotation_batch = actual_relative_rotation_batch.detach().numpy()
        expected_relative_rotation_batch = expected_relative_rotation_batch.detach().numpy()
        numpy.testing.assert_allclose(actual_relative_rotation_batch, expected_relative_rotation_batch, rtol=7.1029973e-07)

    def test_BatchRelativeTranslationMatricesToAbsolute(self):
        input_translation = [1,1,1]
        expected_relative_translation_batch = torch.Tensor([input_translation, input_translation, input_translation, input_translation,
                                                   input_translation, input_translation, input_translation, input_translation]).requires_grad_(True)
        # 2 segments of 4 translations
        expected_relative_translation_batch = expected_relative_translation_batch.view((2, 4, 3))

        absolute_translation_batch = TensorGeometry.batch_assembleDeltaTranslationMatrices(expected_relative_translation_batch)
        self.assertTrue(absolute_translation_batch.requires_grad)
        actual_relative_translation_batch = torch.zeros(expected_relative_translation_batch.shape)

        # take the absolute translations tensor and make the translations relative
        for i, segment in enumerate(absolute_translation_batch):
            for j, translation in enumerate(segment[1:]):
                actual_relative_translation_batch[i, j] = translation - segment[j]

        actual_relative_translation_batch = actual_relative_translation_batch.detach().numpy()
        expected_relative_translation_batch = expected_relative_translation_batch.detach().numpy()
        numpy.testing.assert_allclose(actual_relative_translation_batch, expected_relative_translation_batch)

    def test_batch_computeDeltaTranslations(self):
        input_translation = [1,1,1]
        expected_relative_translation_batch = torch.Tensor([input_translation, input_translation, input_translation, input_translation,
                                                   input_translation, input_translation, input_translation, input_translation]).requires_grad_(True)
        # 2 segments of 4 translations
        expected_relative_translation_batch = expected_relative_translation_batch.view((2, 4, 3))

        global_positions_batch = TensorGeometry.batch_assembleDeltaTranslationMatrices(expected_relative_translation_batch)
        self.assertTrue(global_positions_batch.requires_grad)
        acutal_relative_translation_batch = TensorGeometry.batch_computeDeltaTranslations(global_positions_batch)
        self.assertTrue(acutal_relative_translation_batch.requires_grad)

        expected_relative_translation_batch = expected_relative_translation_batch.detach().numpy()
        acutal_relative_translation_batch = acutal_relative_translation_batch.detach().numpy()
        numpy.testing.assert_allclose(acutal_relative_translation_batch, expected_relative_translation_batch)

    def test_batch_assembleDeltaEulerAngles(self):
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        input_angles = [y, x_prime, z_prime_prime]

        relative_euler_rotation_batch = torch.Tensor(
            [input_angles, input_angles, input_angles, input_angles, input_angles, input_angles, input_angles,
             input_angles]).requires_grad_(True)

        # 2 segments of 4 euler rotations
        relative_euler_rotation_batch = relative_euler_rotation_batch.view((2, 4, 3))

        actual_absolute_euler_orientation_batch = TensorGeometry.batch_assembleDeltaEulerAngles(relative_euler_rotation_batch)
        self.assertTrue(actual_absolute_euler_orientation_batch.requires_grad)
        actual_absolute_matrix_orientation_batch = TensorGeometry.batchEulerAnglesToRotationMatrixTensor(actual_absolute_euler_orientation_batch)
        self.assertTrue(actual_absolute_matrix_orientation_batch.requires_grad)
        actual_relative_rotation_batch = torch.zeros(actual_absolute_matrix_orientation_batch[:, 1:, :].shape)

        #take the absolute rotation tensor and make the rotations relative
        for i, segment in enumerate(actual_absolute_matrix_orientation_batch):
            for j, rotation in enumerate(segment[1:]):
                actual_relative_rotation_batch[i, j] = torch.mm(rotation, segment[j].inverse())

        actual_relative_rotation_batch = TensorGeometry.batchRotationMatrixTensorToEulerAngles(actual_relative_rotation_batch)
        self.assertTrue(actual_relative_rotation_batch.requires_grad)

        actual_relative_rotation_batch = actual_relative_rotation_batch.detach().numpy()
        expected_relative_rotation_batch = relative_euler_rotation_batch.detach().numpy()
        numpy.testing.assert_allclose(actual_relative_rotation_batch, expected_relative_rotation_batch, rtol=7.1029973e-07)

    def test_batchEulerDifferences(self):
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        input_angles = [y, x_prime, z_prime_prime]

        expected_euler_rotation_batch = torch.Tensor(
            [input_angles, input_angles, input_angles, input_angles, input_angles, input_angles, input_angles,
             input_angles]).requires_grad_(True)

        # 2 segments of 4 euler rotations
        expected_euler_rotation_batch = expected_euler_rotation_batch.view((2, 4, 3))

        absolute_euler_orientation_batch = TensorGeometry.batch_assembleDeltaEulerAngles(
            expected_euler_rotation_batch)
        self.assertTrue(absolute_euler_orientation_batch.requires_grad)

        actual_euler_rotation_batch = TensorGeometry.batch_eulerDifferences(absolute_euler_orientation_batch)
        self.assertTrue(actual_euler_rotation_batch.requires_grad)

        numpy.testing.assert_allclose(actual_euler_rotation_batch.detach().numpy(),
                                      expected_euler_rotation_batch.detach().numpy(),
                                      rtol=7.1029973e-07)


class Test_Geometry(TestCase):
    def setUp(self) -> None:
        self.dataframe5x5:pd.DataFrame = pd.read_pickle("../datainfo/test_seq5x5.pickle")
        self.segment = Segment(self.dataframe5x5.iloc[1, :])
        self.dataset = KITTIImageSequenceDatasetEulerDifferences("../datainfo/test_seq5x5.pickle", new_size=(512, 512),
                                                            img_mean=[1, 1, 1], img_std=[1, 1, 1])

    def test_quaternions_to_tait_bryan_rotations(self):
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        tait_bryan = numpy.asarray([y, x_prime, z_prime_prime])
        quat = Geometry.tait_bryan_rotation_to_quaternion(tait_bryan)

        series_of_rotations = numpy.asarray([quat, quat, quat])
        expected = numpy.asarray([tait_bryan, tait_bryan, tait_bryan])

        actual = numpy.asarray(Geometry.quaternions_to_tait_bryan_rotations(series_of_rotations))

        numpy.testing.assert_almost_equal(actual, expected)

    def test_assemble_delta_tait_bryan_rotations(self):
        assembled = Geometry.assemble_delta_tait_bryan_rotations(self.dataset.__getitem__(1)[1].numpy()[:, :3])
        expected = numpy.asarray(Geometry.quaternions_to_tait_bryan_rotations(self.segment.get_attitudes()))
        numpy.testing.assert_almost_equal(assembled, expected)

    def test_assemble_delta_translations(self):
        assembled = Geometry.assemble_delta_translations(self.dataset.__getitem__(1)[1].numpy()[:, 3:])
        expected = numpy.asarray(self.segment.get_positions())
        numpy.testing.assert_almost_equal(assembled, expected)

    def test_assemble_batch_delta_translations(self):
        segment = self.dataset.__getitem__(1)[1].numpy()[:, 3:]
        batch = [segment, segment, segment]
        batch = numpy.asarray(batch)

        assembled_batch = Geometry.assemble_batch_delta_translations(batch)
        expected = numpy.asarray(self.segment.get_positions())
        expected_batch = [expected, expected, expected]
        expected_batch = numpy.asarray(expected_batch)

        numpy.testing.assert_almost_equal(assembled_batch, expected_batch)

    def test_assemble_batch_delta_tait_bryan_rotations(self):
        segment = self.dataset.__getitem__(1)[1].numpy()[:, :3]
        batch = [segment, segment, segment]
        batch = numpy.asarray(batch)

        assembled_batch = Geometry.assemble_batch_delta_tait_bryan_rotations(batch)
        expected = numpy.asarray(Geometry.quaternions_to_tait_bryan_rotations(self.segment.get_attitudes()))

        expected_batch = [expected, expected, expected]
        expected_batch = numpy.asarray(expected_batch)

        numpy.testing.assert_almost_equal(assembled_batch, expected_batch)

    def test_reset_euler_orientations_to_origin(self):
        assembled = Geometry.assemble_delta_tait_bryan_rotations(self.dataset.__getitem__(1)[1].numpy()[:, :3])
        initial_rotation = Geometry.matrix_to_tait_bryan_euler(self.dataframe5x5.iloc[1, 2][0, :9])
        expected = Geometry.rotation_matrices_to_euler(self.dataframe5x5.iloc[1, 2][:, :9])
        resetted = Geometry.reset_euler_orientations_to_origin(initial_rotation, assembled)
        numpy.testing.assert_almost_equal(resetted, expected)

    def test_get_position_differences(self):
        expected = self.dataset.__getitem__(1)[1].numpy()[:, 3:]
        differences = Geometry.get_position_differences(numpy.asarray(self.segment.get_positions()))
        numpy.testing.assert_almost_equal(differences, expected)

    def test_get_tait_bryan_orientation_differences(self):
        expected = self.dataset.__getitem__(1)[1].numpy()[:, :3]
        attitudes = numpy.asarray(Geometry.quaternions_to_tait_bryan_rotations(self.segment.get_attitudes()))
        differences = Geometry.get_tait_bryan_orientation_differences(attitudes)
        numpy.testing.assert_almost_equal(differences, expected)

    def test_tait_bryan_rotation_to_matrix(self):
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        tait_bryan = numpy.asarray([y, x_prime, z_prime_prime])

        matrix = Geometry.tait_bryan_rotation_to_matrix(tait_bryan)
        self.assertEqual(matrix.shape, (3,3))
        numpy.testing.assert_almost_equal(tait_bryan, Rotation.from_matrix(matrix).as_euler("YXZ"))

    def test_tait_bryan_rotations_to_matrices(self):
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        tait_bryans = numpy.asarray([[y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime]])
        matrices = Geometry.tait_bryan_rotation_to_matrix(tait_bryans)
        self.assertEqual(matrices.shape, (3, 3, 3))
        numpy.testing.assert_almost_equal(tait_bryans, Rotation.from_matrix(matrices).as_euler("YXZ"))

    def test_poses_to_transformations_matrix(self):
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        tait_bryans = numpy.asarray([[y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime]])

        x = 1
        y = 2
        z = 3

        locations = numpy.asarray([[x, y, z], [x, y, z], [x, y, z]])
        transformations = Geometry.poses_to_transformations_matrix(locations, tait_bryans)
        self.assertEqual(transformations.shape, (3, 4, 4))
        for transformation in transformations:
            numpy.testing.assert_equal(transformation[3, :], [0,0,0,1])

        numpy.testing.assert_almost_equal(tait_bryans, Rotation.from_matrix(transformations[:, :3, :3]).as_euler("YXZ"))
        numpy.testing.assert_almost_equal(locations, transformations[:, :3, 3])