import shutil
import zipfile
from unittest import TestCase

import h5py
import numpy
import pandas as pd
import torch
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

from Common.Helpers import Geometry
from Datasets.MidAir import MidAirDataSegmenter, MidAirImageSequenceDataset, MidAirImageSequenceDatasetEulerDifferences, \
    Segment, \
    SegmentMapper, MidAirDataPreprocessor


class TestMidAirDataSegmenter(TestCase):
    def setUp(self):
        with zipfile.ZipFile("./Ressources/MidAir.zip", "r") as zip_ref:
            zip_ref.extractall("./Ressources")
        self.processor = MidAirDataPreprocessor("./Ressources/MidAir")
        self.processor.clean()

    def tearDown(self):
        shutil.rmtree("./Ressources/MidAir")

    def test_segment_givenSingleSegmentLength_shouldProduceFixedLengthSegments(self):
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        data_segmenter.segment((4,), 0)
        sensor_records = h5py.File("./Ressources/MidAir/Kite_test/cloudy/sensor_records.hdf5", "r+")

        for trajectory in sensor_records:
            dataframe = pd.DataFrame(sensor_records[trajectory]["trajectory_segments"],
                                     columns=["sequence_length", "start_frame_index"])
            expected_length = len(
                list(range(0, data_segmenter._get_trajectory_length(sensor_records[trajectory]) - 4, 4)))

            self.assertEqual(len(dataframe), expected_length)

            expected_start_Frame_index = 0
            for i, row in dataframe.iterrows():
                self.assertEqual(row["sequence_length"], 4)
                self.assertEqual(row["start_frame_index"], expected_start_Frame_index)
                expected_start_Frame_index += 4

    def test_segment_when_resegmenting_should_overwrite_segment(self):
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        # first segmentation
        data_segmenter.segment((4,), 0)
        # second segmentation
        data_segmenter.segment((6,), 0)
        sensor_records = h5py.File("./Ressources/MidAir/Kite_test/cloudy/sensor_records.hdf5", "r+")

        for trajectory in sensor_records:
            dataframe = pd.DataFrame(sensor_records[trajectory]["trajectory_segments"],
                                     columns=["sequence_length", "start_frame_index"])
            expected_length = len(
                list(range(0, data_segmenter._get_trajectory_length(sensor_records[trajectory]) - 6, 6)))

            self.assertEqual(len(dataframe), expected_length)

            expected_start_Frame_index = 0
            for i, row in dataframe.iterrows():
                self.assertEqual(row["sequence_length"], 6)
                self.assertEqual(row["start_frame_index"], expected_start_Frame_index)
                expected_start_Frame_index += 6

    def test_segment_givenNoneSegmentLength_shouldSegmentWillBeTrajectory(self):
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        data_segmenter.segment(None)
        sensor_records = h5py.File("./Ressources/MidAir/Kite_test/cloudy/sensor_records.hdf5", "r+")

        for trajectory in sensor_records:
            dataframe = pd.DataFrame(sensor_records[trajectory]["trajectory_segments"],
                                     columns=["sequence_length", "start_frame_index"])
            expected_length = 1
            self.assertEqual(len(dataframe), expected_length)

            for i, row in dataframe.iterrows():
                self.assertEqual(row["sequence_length"], data_segmenter
                                 ._get_trajectory_length(sensor_records[trajectory]))
                self.assertEqual(row["start_frame_index"], 0)

    def test_segment_givenSingleSegmentLengthAndOverlap_shouldProduceFixedLengthOverlappingSegments(self):
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        data_segmenter.segment((4,), 1)
        sensor_records = h5py.File("./Ressources/MidAir/Kite_test/cloudy/sensor_records.hdf5", "r+")

        for trajectory in sensor_records:
            dataframe = pd.DataFrame(sensor_records[trajectory]["trajectory_segments"],
                                     columns=["sequence_length", "start_frame_index"])
            expected_length = len(
                list(range(0, data_segmenter._get_trajectory_length(sensor_records[trajectory]) - 4, 3)))

            self.assertEqual(len(dataframe), expected_length)

            expected_start_Frame_index = 0
            for i, row in dataframe.iterrows():
                self.assertEqual(row["sequence_length"], 4)
                self.assertEqual(row["start_frame_index"], expected_start_Frame_index)
                expected_start_Frame_index += 3

    def test_segment_givenRangeSegmentLength_shouldProduceVariableLengthOverlappingSegments(self):
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        data_segmenter.segment((2, 4), 1)
        sensor_records = h5py.File("./Ressources/MidAir/Kite_test/cloudy/sensor_records.hdf5", "r+")

        for trajectory in sensor_records:
            dataframe = pd.DataFrame(sensor_records[trajectory]["trajectory_segments"],
                                     columns=["sequence_length", "start_frame_index"])

            self.assertTrue(3 in dataframe["sequence_length"])
            self.assertTrue(4 in dataframe["sequence_length"])
            self.assertTrue(2 in dataframe["sequence_length"])

            expected_start_Frame_index = 0
            for i, row in dataframe.iterrows():
                self.assertTrue(row["sequence_length"] <= 4 or row["sequence_length"] >= 2)
                self.assertEqual(row["start_frame_index"], expected_start_Frame_index)
                expected_start_Frame_index += row["sequence_length"] - 1

    def test_segment_givenRangeSegmentLength_shouldProduceVariableLengthSegments(self):
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        data_segmenter.segment((2, 4), 0)
        sensor_records = h5py.File("./Ressources/MidAir/Kite_test/cloudy/sensor_records.hdf5", "r+")

        for trajectory in sensor_records:
            dataframe = pd.DataFrame(sensor_records[trajectory]["trajectory_segments"],
                                     columns=["sequence_length", "start_frame_index"])

            self.assertTrue(3 in dataframe["sequence_length"])
            self.assertTrue(4 in dataframe["sequence_length"])
            self.assertTrue(2 in dataframe["sequence_length"])

            expected_start_Frame_index = 0
            for i, row in dataframe.iterrows():
                self.assertTrue(row["sequence_length"] <= 4 or row["sequence_length"] >= 2)
                self.assertEqual(row["start_frame_index"], expected_start_Frame_index)
                expected_start_Frame_index += row["sequence_length"]


class TestMidAirImageSequenceDataset(TestCase):
    def setUp(self):
        with zipfile.ZipFile("./Ressources/MidAir.zip", "r") as zip_ref:
            zip_ref.extractall("./Ressources")
        self.processor = MidAirDataPreprocessor("./Ressources/MidAir")
        self.processor.clean()
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        data_segmenter.segment((4,), 0)

    def tearDown(self):
        shutil.rmtree("./Ressources/MidAir")

    def test_givenDataset_whenCallingLen_ShouldReturnTotalNumberOfSegmentsInTheDatasetPath(self):
        dataset = MidAirImageSequenceDataset("./Ressources/MidAir/Kite_test", new_size=(512, 512))

        total_number_of_segments = 0

        with dataset.HDF5["./Ressources/MidAir/Kite_test/cloudy"] as hdf5:
            for trajectory in hdf5:
                total_number_of_segments += len(
                    hdf5[trajectory]["trajectory_segments"])

        self.assertEqual(total_number_of_segments, dataset.__len__())

    def test_givenSpecificTrajectories_whenCallingLen_ShouldReturnTotalNumberOfSegmentsInTheTrajectoryPath(self):
        dataset = MidAirImageSequenceDataset("./Ressources/MidAir/Kite_test", new_size=(512, 512),
                                             trajectories=["trajectory_3001"])

        with dataset.HDF5["./Ressources/MidAir/Kite_test/cloudy"] as hdf5:
            total_number_of_segments = len(
                    hdf5["trajectory_3001"]["trajectory_segments"])

        self.assertEqual(total_number_of_segments, dataset.__len__())

    def test_givenDataset_whenCallingItem_ShouldReturnTheTupleOfTheCorrespondingSegment(self):
        dataset = MidAirImageSequenceDataset("./Ressources/MidAir/Kite_test", new_size=(512, 512), img_mean=[1, 1, 1], img_std=[1, 1, 1])
        first_segment, img_sequence = dataset.__getitem__(0)

        # First item should be the segment
        self.assertTrue(isinstance(first_segment, Segment))

        # Second item should be the segment's images as image tensors
        self.assertEqual(len(img_sequence), 4)

        # verify the image size and its type
        for image in img_sequence:
            self.assertTrue(isinstance(image, torch.FloatTensor))
            self.assertEqual(image.shape, (3, 512, 512))

    def test_givenDataset_when_no_trajectories_specified_should_map_all_segments(self):
        # re-segment so that 1 trajectory equals 1 segment
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        data_segmenter.segment()

        dataset = MidAirImageSequenceDataset("./Ressources/MidAir/Kite_test", new_size=(512, 512))

        self.assertEqual(dataset.__len__(), 5)

    def test_given_sensor_record_when_trajectories_specified_should_map_segments_of_specified_trajectories_only(self):
        # re-segment so that 1 trajectory equals 1 segment
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        data_segmenter.segment()

        dataset = MidAirImageSequenceDataset("./Ressources/MidAir/Kite_test", new_size=(512, 512),
                                             trajectories=["trajectory_3000", "trajectory_3001"])

        self.assertEqual(dataset.__len__(), 2)
        for i in range(0, dataset.__len__()):
            self.assertTrue(dataset._get_segment(i).trajectory in ["trajectory_3000", "trajectory_3001"])


class TestSegment(TestCase):
    def setUp(self):
        with zipfile.ZipFile("./Ressources/MidAir.zip", "r") as zip_ref:
            zip_ref.extractall("./Ressources")
        self.processor = MidAirDataPreprocessor("./Ressources/MidAir")
        self.processor.clean()
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        data_segmenter.segment((4,), 0)

        self.dataset: MidAirImageSequenceDataset = MidAirImageSequenceDataset("./Ressources/MidAir/Kite_test", new_size=(512, 512),
                                                  img_mean=(1, 1, 1))
        self.second_segment: Segment = self.dataset._get_segment(1)

    def tearDown(self):
        shutil.rmtree("./Ressources/MidAir")

    def test_givenDataset_whenCallingGetPosition_shouldReturnProperPosition(self):
        with self.dataset.HDF5["./Ressources/MidAir/Kite_test/cloudy"] as hdf5:
            expected_position = hdf5["trajectory_3000"]["groundtruth"][
                                    "position"][4:8]
            initial_rotation = Quaternion(hdf5["trajectory_3000"]["groundtruth"][
                                    "attitude"][4])
            initial_position = expected_position[0]

        actual_positions_camera_frame = self.second_segment.get_positions().numpy()
        self.assertEqual(actual_positions_camera_frame.shape, (4, 3))
        numpy.testing.assert_array_equal(actual_positions_camera_frame[0], numpy.asarray([0.0, 0.0, 0.0]))
        #remap axis to original absolute frame
        actual_position = self._rotate_camera_frame_to_world_frame(torch.Tensor(actual_positions_camera_frame)).numpy()

        for i, pos in enumerate(actual_positions_camera_frame):
            #assert that the positions were switch to the KITTI camera coordinate systen
            numpy.testing.assert_array_equal(actual_position[i], pos[[2, 0, 1]])

        actual_position = Geometry.remap_position_axes(initial_rotation, actual_position)
        actual_position = Geometry.reset_positions_to_origin(initial_position, actual_position)
        self.assertTrue(numpy.allclose(actual_position, expected_position))

    def test_givenDataset_whenCallingGetPositionDifferences_shouldReturnProperPositionDifferences(self):
        with self.dataset.HDF5["./Ressources/MidAir/Kite_test/cloudy"] as hdf5:
            expected_position = hdf5["trajectory_3000"]["groundtruth"][
                                    "position"][4:8]
            initial_rotation = Quaternion(hdf5["trajectory_3000"]["groundtruth"][
                                    "attitude"][4])
            initial_position = expected_position[0]

        #Reverse position differences
        actual_position_differences = self.second_segment.get_position_differences().numpy()
        self.assertEqual(actual_position_differences.shape, (3, 3))
        actual_positions_camera_frame = Geometry.assemble_delta_translations(actual_position_differences)

        #remap axis to original absolute frame
        actual_positions = self._rotate_camera_frame_to_world_frame(torch.Tensor(actual_positions_camera_frame)).numpy()

        for i, pos in enumerate(actual_positions_camera_frame):
            #assert that the positions were switch to the KITTI camera coordinate systen
            numpy.testing.assert_array_equal(actual_positions[i], pos[[2, 0, 1]])

        actual_positions = Geometry.remap_position_axes(initial_rotation, actual_positions)
        actual_positions = Geometry.reset_positions_to_origin(initial_position, actual_positions)

        self.assertEqual(actual_positions.shape, (4, 3))
        self.assertTrue(numpy.allclose(actual_positions, expected_position))

    def test_givenDataset_whenCallingGetAttitudes_shouldReturnProperAttitude(self):
        with self.dataset.HDF5["./Ressources/MidAir/Kite_test/cloudy"] as hdf5:
            expected_attitude = hdf5["trajectory_3000"]["groundtruth"][
                                    "attitude"][4:8]
            initial_attitude = expected_attitude[0]

        actual_attitude = self.second_segment.get_attitudes()
        self.assertEqual(len(actual_attitude), 4)
        self.assertEqual(None, numpy.testing.assert_array_almost_equal(actual_attitude[0].elements,
                                                                numpy.asarray([1.0, 0.0, 0.0, 0.0])))
        actual_attitude_camera_frame = Geometry.quaternion_elements_to_quaternions(actual_attitude)
        actual_attitude = self._rotate_quaternions_to_world_frame(actual_attitude_camera_frame)

        for i, quat in enumerate(actual_attitude_camera_frame):
            imaginaries = quat.elements[1:]
            #assert that the quaternions were switch to the KITTI camera coordinate systen
            numpy.testing.assert_array_equal(actual_attitude[i].elements[1:], imaginaries[[2, 0, 1]])

        actual_attitude = Geometry.reset_orientations_to_origin(Quaternion(initial_attitude), actual_attitude)

        self.assertTrue(numpy.allclose(numpy.asarray(
                                           [att.elements for att in actual_attitude]
                                       ),
                                           expected_attitude
                                       ))

    def test_givenDataset_whenCallingGetAttitudeDifferences_shouldReturnProperAttitudeDifference(self):
        with self.dataset.HDF5["./Ressources/MidAir/Kite_test/cloudy"] as hdf5:
            expected_attitude = hdf5["trajectory_3000"]["groundtruth"][
                                    "attitude"][4:8]
            initial_attitude = expected_attitude[0]

        actual_attitude_differences = self.second_segment.get_attitude_differences()
        self.assertEqual(len(actual_attitude_differences), 3)
        self.assertFalse(numpy.allclose(actual_attitude_differences[0].elements,
                                                                numpy.asarray([1.0, 0.0, 0.0, 0.0])))

        actual_attitude = Geometry.assemble_delta_quaternion_rotations(actual_attitude_differences)

        self.assertEqual(len(actual_attitude), 4)
        self.assertEqual(None, numpy.testing.assert_array_almost_equal(actual_attitude[0].elements,
                                                                numpy.asarray([1.0, 0.0, 0.0, 0.0])))
        actual_attitude_camera_frame = Geometry.quaternion_elements_to_quaternions(actual_attitude)
        actual_attitude = self._rotate_quaternions_to_world_frame(actual_attitude_camera_frame)

        for i, quat in enumerate(actual_attitude_camera_frame):
            imaginaries = quat.elements[1:]
            #assert that the quaternions were switch to the KITTI camera coordinate systen
            numpy.testing.assert_array_equal(actual_attitude[i].elements[1:], imaginaries[[2, 0, 1]])

        actual_attitude = Geometry.reset_orientations_to_origin(Quaternion(initial_attitude), actual_attitude)

        self.assertTrue(numpy.allclose(numpy.asarray(
                                           [att.elements for att in actual_attitude]
                                       ),
                                           expected_attitude
                                       ))

    def test_rotate_quaternion_to_camera_frame(self):
        pose_deg = numpy.array([1, 1.5, 1.2])
        # Degrees to radians
        pose_quat = Quaternion(Rotation.from_euler("ZYX", pose_deg).as_quat()[[3, 0, 1, 2]])
        segment = Segment(root="", trajectory="", camera_view="", start_frame_index=0, segment_length=1, hdf5=None)

        remapped_quat = segment._rotate_quaternion_to_camera_frame(pose_quat)
        pose_deg_unmapped_ZYX = Rotation.from_quat(pose_quat.elements[[1, 2, 3, 0]]).as_euler("ZYX")
        pose_deg_unmapped_YXZ = Rotation.from_quat(pose_quat.elements[[1, 2, 3, 0]]).as_euler("YXZ")
        pose_deg_remapped_YXZ = Rotation.from_quat(remapped_quat.elements[[1, 2, 3, 0]]).as_euler("YXZ")

        self.assertNotEqual(pose_deg_unmapped_ZYX[0], pose_deg_unmapped_YXZ[0])
        self.assertNotEqual(pose_deg_unmapped_ZYX[1], pose_deg_unmapped_YXZ[1])
        self.assertNotEqual(pose_deg_unmapped_ZYX[2], pose_deg_unmapped_YXZ[2])
        numpy.testing.assert_almost_equal(pose_deg_unmapped_ZYX, pose_deg)

        # Remaped quaternion should have yaw-pitch-roll axis as YXZ respectively
        numpy.testing.assert_array_equal(pose_deg_unmapped_ZYX, pose_deg_remapped_YXZ)

    def test_remap_position_axis_to_camera_frame(self):
        pose = numpy.array([[1, 1.5, 1.2], [1, 1.5, 1.2]])
        segment = Segment(root="", trajectory="", camera_view="", start_frame_index=0, segment_length=1, hdf5=None)

        remapped_pose = segment._rotate_world_frame_to_camera_frame(torch.Tensor(pose)).numpy()

        numpy.testing.assert_almost_equal(remapped_pose, [[1.5, 1.2, 1], [1.5, 1.2, 1]])

    def _rotate_quaternions_to_world_frame(self, quatertions):
        # Rotate the i,j,k (x,y,z) component of the quaternion to the camera frame using the rotation matrix
        remapped_quaternions = []
        for quaternion in quatertions:
            world_frame_quat = numpy.append(quaternion.elements[:1],
                                             quaternion.elements[1:].dot(
                                                 self._get_to_world_frame_rotation_matrix()))
            remapped_quaternions.append(Quaternion(world_frame_quat))
        return remapped_quaternions

    def _rotate_camera_frame_to_world_frame(self, position: torch.Tensor) -> torch.Tensor:
        return torch.mm(position, torch.Tensor(self._get_to_world_frame_rotation_matrix()))

    def _get_to_world_frame_rotation_matrix(self) -> numpy.ndarray:
        """
        Rotation matrix describing the tranformation to the world frame of midair (x: forward, y:right, z:down)
        from the camera frame (x: right, y:down, z:forward)
        @return:
        """
        to_camera_frame_rotation_matrix = numpy.asarray([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]])
        return to_camera_frame_rotation_matrix


class TestMidAirImageSequenceDatasetDeepVO(TestCase):
    def setUp(self):
        with zipfile.ZipFile("./Ressources/MidAir.zip", "r") as zip_ref:
            zip_ref.extractall("./Ressources")
        self.processor = MidAirDataPreprocessor("./Ressources/MidAir")
        self.processor.clean()
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test")
        data_segmenter.segment((4,), 0)

    def tearDown(self):
        shutil.rmtree("./Ressources/MidAir")

    def test_givenDataset_whenCallingItem_ShouldReturnTheTupleOfTheCorrespondingSegment(self):
        dataset = MidAirImageSequenceDatasetEulerDifferences("./Ressources/MidAir/Kite_test", new_size=(512, 512),
                                                             img_mean=[1, 1, 1], img_std=[1, 1, 1],
                                                             trajectories=["trajectory_3000"])
        first_segment = dataset.__getitem__(1)

        # First item should be the segment's images as image tensors
        self.assertEqual(len(first_segment[0]), 4)

        # verify the image size and its type
        for image in first_segment[0]:
            self.assertTrue(isinstance(image, torch.FloatTensor))
            self.assertEqual(image.shape, (3, 512, 512))

        # Third item should be the image's pose
        self.assertEqual(first_segment[1].shape, (3, 6))

    def test_givenDataset_whenCallingItem_ShouldReturnTheTupleWithProperAttitude(self):
        dataset = MidAirImageSequenceDatasetEulerDifferences("./Ressources/MidAir/Kite_test", new_size=(512, 512),
                                                             img_mean=[1, 1, 1], img_std=[1, 1, 1],
                                                             trajectories=["trajectory_3000"])
        relative_attitude = dataset.__getitem__(1)[1][:, :3]

        with dataset.HDF5["./Ressources/MidAir/Kite_test/cloudy"] as hdf5:
            expected_attitude = hdf5["trajectory_3000"]["groundtruth"][
                                    "attitude"][4:8]
            initial_rotation = Quaternion(expected_attitude[0])

        relative_attitude = Geometry.tait_bryan_rotations_to_quaternions(relative_attitude)
        actual_attitudes = Geometry.assemble_delta_quaternion_rotations(relative_attitude)
        actual_attitudes = self._rotate_quaternions_to_world_frame(actual_attitudes)

        actual_attitudes = Geometry.reset_orientations_to_origin(initial_rotation, actual_attitudes)
        actual_attitudes = [att.elements for att in actual_attitudes]
        self.assertTrue(numpy.allclose(actual_attitudes, expected_attitude))

    def test_givenDataset_whenCallingItem_ShouldReturnTheTupleWithProperPosition(self):
        dataset = MidAirImageSequenceDatasetEulerDifferences("./Ressources/MidAir/Kite_test", new_size=(512, 512),
                                                             img_mean=[1, 1, 1], img_std=[1, 1, 1],
                                                             trajectories=["trajectory_3000"])

        with dataset.HDF5["./Ressources/MidAir/Kite_test/cloudy"] as hdf5:
            expected_position = hdf5["trajectory_3000"]["groundtruth"][
                                    "position"][4:8]
            initial_rotation = Quaternion(hdf5["trajectory_3000"]["groundtruth"][
                                    "attitude"][4])
            initial_position = expected_position[0]

        #Reverse position differences
        actual_position_differences = dataset.__getitem__(1)[1][:, 3:].numpy()
        self.assertEqual(actual_position_differences.shape, (3, 3))

        actual_positions = Geometry.assemble_delta_translations(actual_position_differences)
        actual_positions = self._rotate_camera_frame_to_world_frame(torch.Tensor(actual_positions)).numpy()
        #remap axis to original absolute frame
        actual_positions = Geometry.remap_position_axes(initial_rotation, actual_positions)
        actual_positions = Geometry.reset_positions_to_origin(initial_position, actual_positions)


        self.assertEqual(actual_positions.shape, (4, 3))
        self.assertTrue(numpy.allclose(actual_positions, expected_position))

    def _rotate_camera_frame_to_world_frame(self, position: torch.Tensor) -> torch.Tensor:
        return torch.mm(position, torch.Tensor(self._get_to_world_frame_rotation_matrix()))

    def _rotate_quaternions_to_world_frame(self, quatertions):
        # Rotate the i,j,k (x,y,z) component of the quaternion to the camera frame using the rotation matrix
        remapped_quaternions = []
        for quaternion in quatertions:
            world_frame_quat = numpy.append(quaternion.elements[:1],
                                             quaternion.elements[1:].dot(
                                                 self._get_to_world_frame_rotation_matrix()))
            remapped_quaternions.append(Quaternion(world_frame_quat))
        return remapped_quaternions

    def _get_to_world_frame_rotation_matrix(self) -> numpy.ndarray:
        """
        Rotation matrix describing the tranformation to the world frame of midair (x: forward, y:right, z:down)
        from the camera frame (x: right, y:down, z:forward)
        @return:
        """
        to_camera_frame_rotation_matrix = numpy.asarray([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]])
        return to_camera_frame_rotation_matrix


class TestSegmentMapper(TestCase):
    def setUp(self):
        with zipfile.ZipFile("./Ressources/MidAir.zip", "r") as zip_ref:
            zip_ref.extractall("./Ressources")

        self.processor = MidAirDataPreprocessor("./Ressources/MidAir/Kite_test/")
        self.processor.clean()
        data_segmenter = MidAirDataSegmenter("./Ressources/MidAir/Kite_test/")
        # segment so that 1 trajectory equals 1 segment
        data_segmenter.segment()

        self.sensor_record = h5py.File("./Ressources/MidAir/Kite_test/cloudy/sensor_records.hdf5", "r+")

    def tearDown(self):
        shutil.rmtree("./Ressources/MidAir")

    def test_given_sensor_record_when_no_trajectories_specified_should_map_all_segments(self):
        mapper = SegmentMapper()
        segments = mapper.map_all("./Ressources/MidAir_segment_mapper_tests/MidAir/Kite_test/cloudy/", self.sensor_record)
        self.assertEqual(len(segments), 5)

    def test_given_sensor_record_when_trajectories_specified_should_map_segments_of_specified_trajectories_only(self):
        mapper = SegmentMapper()
        segments = mapper.map_all("./Ressources/MidAir_segment_mapper_tests/MidAir/Kite_test/cloudy/",
                                  self.sensor_record, ["trajectory_3000", "trajectory_3001"])
        self.assertEqual(len(segments), 2)

        for segments in segments:
            self.assertTrue(segments.trajectory in ["trajectory_3000", "trajectory_3001"])