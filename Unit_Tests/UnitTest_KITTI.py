from unittest import TestCase

import PIL
import numpy
import pandas as pd
import torch
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

from Common.Helpers import Geometry
from Datasets.KITTI import Segment, KITTIImageSequenceDataset, KITTIImageSequenceDatasetEulerDifferences


class TestSegment(TestCase):

    def setUp(self) -> None:
        self.dataframe5x5:pd.DataFrame = pd.read_pickle("./datainfo/test_seq5x5.pickle")
        self.dataframe5x5.reset_index(drop=True, inplace=True)
        self.dataframe5x5.to_pickle("./datainfo/test_seq5x5.pickle")
        self.segment = Segment(self.dataframe5x5.iloc[1, :])

    def test_givenDataset_whenCallingGetPosition_shouldReturnProperPosition(self):
        expected_position = self.dataframe5x5.iloc[1, :].pose[:, 9:]
        initial_rotation = Geometry.matrix_to_quaternion(self.dataframe5x5.iloc[1, :].pose[0, :9])
        initial_position = expected_position[0]

        actual_position = self.segment.get_positions().numpy()
        self.assertEqual(actual_position.shape, (5, 3))
        numpy.testing.assert_array_almost_equal(actual_position[0], numpy.asarray([0.0, 0.0, 0.0]))
        #remap axis to original absolute frame
        actual_position = Geometry.remap_position_axes(initial_rotation, actual_position)
        actual_position = Geometry.reset_positions_to_origin(initial_position, actual_position)
        self.assertTrue(numpy.allclose(actual_position, expected_position))

    def test_givenDataset_whenCallingGetPositionDifferences_shouldReturnProperPositionDifferences(self):
        expected_position = self.dataframe5x5.iloc[1, :].pose[:, 9:]
        initial_rotation = Geometry.matrix_to_quaternion(self.dataframe5x5.iloc[1, :].pose[0, :9])
        initial_position = expected_position[0]

        #Reverse position differences
        actual_position_differences = self.segment.get_position_differences().numpy()
        self.assertEqual(actual_position_differences.shape, (4, 3))
        actual_positions = Geometry.assemble_delta_translations(actual_position_differences)

        #remap axis to original absolute frame
        actual_positions = Geometry.remap_position_axes(initial_rotation, actual_positions)
        actual_positions = Geometry.reset_positions_to_origin(initial_position, actual_positions)

        self.assertEqual(actual_positions.shape, (5, 3))
        self.assertTrue(numpy.allclose(actual_positions, expected_position))

    def test_givenDataset_whenCallingGetAttitudes_shouldReturnProperAttitude(self):

        expected_attitude = self.dataframe5x5.iloc[1, :].pose[:, :9]
        initial_attitude = Geometry.matrix_to_quaternion(self.dataframe5x5.iloc[1, :].pose[0, :9])

        actual_attitude = self.segment.get_attitudes()
        self.assertEqual(len(actual_attitude), 5)
        self.assertEqual(None, numpy.testing.assert_array_almost_equal(actual_attitude[0].elements,
                                                                numpy.asarray([1.0, 0.0, 0.0, 0.0])))

        actual_attitude = Geometry.reset_orientations_to_origin(initial_attitude, actual_attitude)

        self.assertTrue(numpy.allclose(numpy.asarray(
                                           [Geometry.quaternion_to_matrix(att) for att in actual_attitude]
                                       ),
                                           expected_attitude
                                       ))

    def test_givenDataset_whenCallingGetAttitudeDifferences_shouldReturnProperAttitudeDifference(self):
        expected_attitude = self.dataframe5x5.iloc[1, :].pose[:, :9]
        initial_attitude = Geometry.matrix_to_quaternion(self.dataframe5x5.iloc[1, :].pose[0, :9])

        actual_attitude_differences = self.segment.get_attitude_differences()
        self.assertEqual(len(actual_attitude_differences), 4)
        self.assertFalse(numpy.allclose(actual_attitude_differences[0].elements,
                                                                numpy.asarray([1.0, 0.0, 0.0, 0.0])))

        actual_attitude = Geometry.assemble_delta_quaternion_rotations(actual_attitude_differences)

        self.assertEqual(len(actual_attitude), 5)
        self.assertEqual(None, numpy.testing.assert_array_almost_equal(actual_attitude[0].elements,
                                                                numpy.asarray([1.0, 0.0, 0.0, 0.0])))

        actual_attitude = Geometry.reset_orientations_to_origin(Quaternion(initial_attitude), actual_attitude)

        self.assertTrue(numpy.allclose(numpy.asarray(
                                           [Geometry.quaternion_to_matrix(att) for att in actual_attitude]
                                       ),
                                           expected_attitude
                                       ))

    def test_givenSegment_whenCallingGetImages_ListOfImages(self):
        for img in self.segment.get_images():
            self.assertIsNotNone(img)
            self.assertTrue(isinstance(img, PIL.Image.Image))


class TestKITTIImageSequenceDataset(TestCase):
    def setUp(self) -> None:
        self.dataframe5x5: pd.DataFrame = pd.read_pickle("./datainfo/test_seq5x5.pickle")
        self.segment = Segment(self.dataframe5x5.iloc[0, :])

    def test_givenDataset_whenCallingLen_ShouldReturnTotalNumberOfSegmentsInTheDatasetPath(self):
        dataset = KITTIImageSequenceDataset("./datainfo/test_seq5x5.pickle", new_size=(512, 512))

        self.assertEqual(len(self.dataframe5x5), dataset.__len__())

    def test_givenDataset_whenCallingItem_ShouldReturnTheTupleOfTheCorrespondingSegment(self):
        dataset = KITTIImageSequenceDataset("./datainfo/test_seq5x5.pickle", new_size=(512, 512), img_mean=[1, 1, 1], img_std=[1, 1, 1])
        first_segment, img_sequence = dataset.__getitem__(0)

        # First item should be the segment
        self.assertTrue(isinstance(first_segment, Segment))

        # Second item should be the segment's images as image tensors
        self.assertEqual(len(img_sequence), 5)

        # verify the image size and its type
        for image in img_sequence:
            self.assertTrue(isinstance(image, torch.FloatTensor))
            self.assertEqual(image.shape, (3, 512, 512))


class TestKITTISequenceDatasetDeepVO(TestCase):
    def setUp(self):
        self.dataframe5x5: pd.DataFrame = pd.read_pickle("./datainfo/test_seq5x5.pickle")
        self.segment = Segment(self.dataframe5x5.iloc[0, :])

    def test_givenDataset_whenCallingItem_ShouldReturnTheTupleOfTheCorrespondingSegment(self):
        dataset = KITTIImageSequenceDatasetEulerDifferences("./datainfo/test_seq5x5.pickle", new_size=(512, 512),
                                                            img_mean=[1, 1, 1], img_std=[1, 1, 1])
        first_segment = dataset.__getitem__(1)

        # First item should be the segment's images as image tensors
        self.assertEqual(len(first_segment[0]), 5)

        # verify the image size and its type
        for image in first_segment[0]:
            self.assertTrue(isinstance(image, torch.FloatTensor))
            self.assertEqual(image.shape, (3, 512, 512))

        # Second item should be the image's pose
        self.assertEqual(first_segment[1].shape, (4, 6))

    def test_givenDataset_whenCallingItem_ShouldReturnTheTupleWithProperAttitude(self):
        dataset = KITTIImageSequenceDatasetEulerDifferences("./datainfo/test_seq5x5.pickle", new_size=(512, 512),
                                                            img_mean=[1, 1, 1], img_std=[1, 1, 1])
        relative_attitude = dataset.__getitem__(0)[1][:, :3]

        expected_attitude = self.dataframe5x5.iloc[0, :].pose[:, :9]
        initial_attitude = Geometry.matrix_to_quaternion(self.dataframe5x5.iloc[0, :].pose[0, :9])

        actual_attitudes = [Quaternion()]
        for i, rotation in enumerate(relative_attitude):
            quat = Rotation.from_euler("YXZ", rotation).as_quat()[[3, 0, 1, 2]]
            quat = Quaternion(quat)
            quat = quat * actual_attitudes[i]
            actual_attitudes.append(quat)

        actual_attitudes = Geometry.reset_orientations_to_origin(Quaternion(initial_attitude), actual_attitudes)
        actual_attitudes = numpy.asarray([Geometry.quaternion_to_matrix(att) for att in actual_attitudes])
        self.assertTrue(numpy.allclose(actual_attitudes, expected_attitude, atol=1.e-7))

    def test_givenDataset_whenCallingItem_ShouldReturnTheTupleWithProperPosition(self):
        dataset = KITTIImageSequenceDatasetEulerDifferences("./datainfo/test_seq5x5.pickle", new_size=(512, 512),
                                                            img_mean=[1, 1, 1], img_std=[1, 1, 1])

        expected_position = self.dataframe5x5.iloc[0, :].pose[:, 9:]
        initial_rotation = Geometry.matrix_to_quaternion(self.dataframe5x5.iloc[0, :].pose[0, :9])
        initial_position = expected_position[0]

        #Reverse position differences
        actual_position_differences = dataset.__getitem__(0)[1][:, 3:].numpy()
        self.assertEqual(actual_position_differences.shape, (4, 3))
        actual_positions = numpy.zeros((5, 3))
        for i, position_difference in enumerate(actual_position_differences):
            actual_positions[i + 1] = actual_positions[i] + position_difference

        #remap axis to original absolute frame
        actual_positions = Geometry.remap_position_axes(initial_rotation, actual_positions)
        actual_positions = Geometry.reset_positions_to_origin(initial_position, actual_positions)

        self.assertEqual(actual_positions.shape, (5, 3))
        self.assertTrue(numpy.allclose(actual_positions, expected_position))