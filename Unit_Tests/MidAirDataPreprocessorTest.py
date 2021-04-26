import os
import pickle
import shutil
import unittest
import zipfile

import h5py
import numpy

from Datasets.MidAir import MidAirDataPreprocessor


class MidAirDataPreprocessorTest(unittest.TestCase):
    def setUp(self):
        with zipfile.ZipFile("./Ressources/MidAir.zip", "r") as zip_ref:
            zip_ref.extractall("./Ressources")
        self.processor = MidAirDataPreprocessor("./Ressources/MidAir")

    def tearDown(self):
        shutil.rmtree("./Ressources/MidAir")

    def test_given_midAirDataset_whenCallingClean_ShouldExtractFramesAndDeleteArchive(self):
        self.processor.clean()
        self.assertTrue(os.path.exists("./Ressources/MidAir/Kite_test/cloudy/color_left/trajectory_3000/000000.JPEG"))
        self.assertFalse(os.path.exists("./Ressources/MidAir/Kite_test/cloudy/color_left/trajectory_3000/frames.zip"))

    def test_given_midAirDataset_whenCallingClean_ShouldExtractSensorRecordAndDeleteArchive(self):
        self.processor.clean()
        self.assertTrue(os.path.exists("./Ressources/MidAir/Kite_test/cloudy/sensor_records.hdf5"))
        self.assertTrue(os.path.exists("./Ressources/MidAir/Kite_test/cloudy/sensor_records.zip"))

    def test_givenMidAirDataset_whenCallingClean_shouldOnlyKeepPositionAndAttitude(self):
        self.processor.clean()
        sensor_records = h5py.File("./Ressources/MidAir/Kite_test/cloudy/sensor_records.hdf5", "r+")
        self.assertFalse("velocity" in sensor_records["trajectory_3000"]["groundtruth"])
        self.assertFalse("acceleration" in sensor_records["trajectory_3000"]["groundtruth"])
        self.assertFalse("angular_velocity" in sensor_records["trajectory_3000"]["groundtruth"])
        self.assertTrue("attitude" in sensor_records["trajectory_3000"]["groundtruth"])
        self.assertTrue("position" in sensor_records["trajectory_3000"]["groundtruth"])

    def test_givenMidAirDataset_whenCallingClean_shouldReduceGroundTruthFrom100HzTo25Hz(self):
        sensor_records_untouched = h5py.File("./Ressources/MidAir/Kite_test/cloudy/sensor_records_untouched.hdf5", "r+")
        self.processor.clean()
        sensor_records = h5py.File("./Ressources/MidAir/Kite_test/cloudy/sensor_records.hdf5", "r+")
        self.assertEqual(sensor_records["trajectory_3000"]["groundtruth"]["position"].len(),
                         sensor_records["trajectory_3000"]["camera_data"]["color_left"].len())
        self.assertEqual(sensor_records["trajectory_3000"]["groundtruth"]["attitude"].len(),
                         sensor_records["trajectory_3000"]["camera_data"]["color_left"].len())

        # select every frame at every start of a second
        hundredHz1Sec_pos = sensor_records_untouched["trajectory_3000"]["groundtruth"]["position"][0::100]
        twentyfiveHz1Sec_pos = sensor_records["trajectory_3000"]["groundtruth"]["position"][0::25]

        hundredHz1Sec_att = sensor_records_untouched["trajectory_3000"]["groundtruth"]["attitude"][0::100]
        twentyfiveHz1Sec_att = sensor_records["trajectory_3000"]["groundtruth"]["attitude"][0::25]

        self.assertTrue(numpy.array_equal(hundredHz1Sec_pos, twentyfiveHz1Sec_pos))
        self.assertTrue(numpy.array_equal(hundredHz1Sec_att, twentyfiveHz1Sec_att))

    def test_givenMidAirDataset_whenCallingComputingMeanStd_shouldCreateFileWithValidValues(self):
        processor = MidAirDataPreprocessor("./Ressources/MidAir_mean_std_test")
        processor.compute_dataset_image_mean_std()

        mean = pickle.load(open("./Ressources/MidAir_mean_std_test/Means.pkl", "rb"))
        std = pickle.load(open("./Ressources/MidAir_mean_std_test/StandardDevs.pkl", "rb"))

        self.assertSequenceEqual(mean["mean_np"], [1.0, 1.0, 1.0])
        self.assertSequenceEqual(mean["mean_tensor"], [0.003921568859368563, 0.003921568859368563, 0.003921568859368563])

        self.assertSequenceEqual(std["std_np"], [0.0, 0.0, 0.0])
        self.assertSequenceEqual(std["std_tensor"], [0.0, 0.0, 0.0])

    def test_givenMidAirDataset_whenCallingComputingMeanStdMinus0point5_shouldCreateFileWithValidValues(self):
        processor = MidAirDataPreprocessor("./Ressources/MidAir_mean_std_test")
        processor.compute_dataset_image_mean_std(True)

        mean = pickle.load(open("./Ressources/MidAir_mean_std_test/Means.pkl", "rb"))
        std = pickle.load(open("./Ressources/MidAir_mean_std_test/StandardDevs.pkl", "rb"))

        self.assertSequenceEqual(mean["mean_np"], [1.0, 1.0, 1.0])
        self.assertSequenceEqual(mean["mean_tensor"], [-0.4960784316062927, -0.4960784316062927, -0.4960784316062927])

        self.assertSequenceEqual(std["std_np"], [0.0, 0.0, 0.0])
        self.assertSequenceEqual(std["std_tensor"], [0.0, 0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
