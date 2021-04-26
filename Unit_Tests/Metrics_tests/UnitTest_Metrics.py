import math
import pickle
from unittest import TestCase

import numpy

from Common.Helpers import Geometry
from Metrics.Metrics import TranslationRotationDrift, AbsoluteTrajectoryError, RelativeError, \
    CompoundTranslationRotationDrift, CompoundAbsoluteTrajectoryError, CompoundRelativeError, MetricLogger, \
    TrajectoryStats


class TestTrajectoryStats(TestCase):
    def setUp(self) -> None:

        self.locations = numpy.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]])

    def test_given_pose_and_differences_should_return_proper_stats(self):
        stats = TrajectoryStats("test_dataset",
                                "test_trajectory",
                                "test_model",
                                self.locations,
                                3)
        self.assertEqual(stats.metrics["framerate"], 3)
        self.assertEqual(stats.metrics["trajectory_lenght"], 2)
        self.assertEqual(stats.metrics["avg_trajectory_speed"], 2)
        self.assertEqual(stats.metrics["x_displacement_range"], (0,2))
        self.assertEqual(stats.metrics["y_displacement_range"], (0,0))
        self.assertEqual(stats.metrics["z_displacement_range"], (0,0))


class TestTranslationRotationDrift(TestCase):
    def setUp(self) -> None:
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        self.tait_bryans = numpy.asarray(
            [[y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime]])

        self.locations = numpy.asarray([[0, 0, 0], [100, 0, 0], [200, 0, 0]])

    def test_Given_same_pose_and_poseGT_sequence_errors_should_be_0(self):
        drift = TranslationRotationDrift("test_dataset", "test_trajectory", "test_model", self.locations,
                                         self.tait_bryans, self.locations, self.tait_bryans,
                                         framerate=1, stepsize=1, align=False)

        self.assertEqual(len(drift.metrics["segment_errors"].keys()), 1)
        self.assertTrue(100 in drift.metrics["segment_errors"].keys())
        self.assertEqual(len(drift.metrics["segment_errors"][100]), 1)
        self.assertEqual(drift.metrics["segment_errors"][100][0]["start_frame"], 0)
        self.assertAlmostEqual(drift.metrics["segment_errors"][100][0]["rotation_error_percent"], 0)
        self.assertAlmostEqual(drift.metrics["segment_errors"][100][0]["translation_error_percent"], 0)
        self.assertEqual(drift.metrics["segment_errors"][100][0]["segment_lenght"], 100)
        self.assertAlmostEqual(drift.metrics["segment_errors"][100][0]["speed"], 33.3333333)
        self.assertAlmostEqual(drift.metrics["average_segment_errors"][100]["average_rotation_error"], 0)
        self.assertAlmostEqual(drift.metrics["average_segment_errors"][100]["average_translation_error"], 0)
        self.assertEqual(len(drift.metrics["average_segment_errors"][200]), 0)
        self.assertEqual(len(drift.metrics["average_segment_errors"][300]), 0)
        self.assertEqual(len(drift.metrics["average_segment_errors"][400]), 0)
        self.assertEqual(len(drift.metrics["average_segment_errors"][500]), 0)
        self.assertEqual(len(drift.metrics["average_segment_errors"][600]), 0)
        self.assertEqual(len(drift.metrics["average_segment_errors"][700]), 0)
        self.assertEqual(len(drift.metrics["average_segment_errors"][800]), 0)
        self.assertAlmostEqual(drift.metrics["avg_translation_error"], 0)
        self.assertAlmostEqual(drift.metrics["avg_rotation_error_radians"], 0)
        self.assertAlmostEqual(drift.metrics["avg_translation_error_percent"], 0)
        self.assertAlmostEqual(drift.metrics["avg_rotation_error_degrees_per_meter"], 0)

    def test_given_valid_poses_should_return_complete_metric(self):
        drift = TranslationRotationDrift("test_dataset", "test_trajectory", "test_model", self.locations,
                                         self.tait_bryans, self.locations, self.tait_bryans,
                                         framerate=1, stepsize=1, align=False)

        self.assertEqual(drift.metrics["dataset"], "test_dataset")
        self.assertEqual(drift.metrics["trajectory"], "test_trajectory")
        self.assertEqual(drift.metrics["model"], "test_model")
        self.assertEqual(drift.metrics["framerate"], 1)
        self.assertEqual(drift.metrics["stepsize"], 1)
        self.assertTrue("segment_errors" in drift.metrics.keys())
        self.assertTrue("average_segment_errors" in drift.metrics.keys())
        self.assertTrue("avg_translation_error" in drift.metrics.keys())
        self.assertTrue("avg_rotation_error_radians" in drift.metrics.keys())
        self.assertTrue("avg_translation_error_percent" in drift.metrics.keys())
        self.assertTrue("avg_rotation_error_degrees_per_meter" in drift.metrics.keys())
        with drift.figures_loader as figures:
            self.assertTrue("average_segment_error_plot" in figures.keys())

    def test_given_kitti_07_poses_should_return_same_estimate_than_original_code_from_git(self):
        with open("kitti_07_location.pkl", "rb") as file:
            locations = pickle.load(file)
        with open("kitti_07_location_gt.pkl", "rb") as file:
            locations_gt = pickle.load(file)
        with open("kitti_07_orientation.pkl", "rb") as file:
            orientations = pickle.load(file)
        with open("kitti_07_orientation_gt.pkl", "rb") as file:
            orientations_gt = pickle.load(file)

        with open("kitti_07_seq_kitti_odom_eval_seq_err.pkl", "rb") as file:
            eval_seq_err = pickle.load(file)

        with open("kitti_07_seq_kitti_odom_eval_avg_segment_errs.pkl", "rb") as file:
            avg_segment_errs = pickle.load(file)

        with open("kitti_07_seq_kitti_odom_eval_overall_er.pkl", "rb") as file:
            overall_er = pickle.load(file)

        with open("kitti_07_seq_kitti_odom_eval_expected_pose_gt.pkl", "rb") as file:
            expected_pose_gt = pickle.load(file)

        with open("kitti_07_seq_kitti_odom_eval_expected_pose.pkl", "rb") as file:
            expected_pose = pickle.load(file)

        for i in expected_pose_gt.keys():
            pose_gt = Geometry.poses_to_transformations_matrix(locations_gt, orientations_gt)
            numpy.testing.assert_almost_equal(expected_pose_gt[i], pose_gt[i])

            pose = Geometry.poses_to_transformations_matrix(locations, orientations)
            numpy.testing.assert_almost_equal(expected_pose[i], pose[i])

        drift = TranslationRotationDrift("test_dataset", "test_trajectory", "test_model", locations_gt,
                                         orientations_gt, locations, orientations,
                                         framerate=10, stepsize=10, align=False)

        numpy.testing.assert_almost_equal(drift.seq_errors, eval_seq_err)

        for i in avg_segment_errs.keys():
            if i == 700 or i == 800:
                continue
            self.assertAlmostEqual(avg_segment_errs[i]['average_translation_error'],
                                   drift.metrics["average_segment_errors"][i]['average_translation_error'])
            self.assertAlmostEqual(avg_segment_errs[i]['average_rotation_error'],
                                   drift.metrics["average_segment_errors"][i]['average_rotation_error'])

        self.assertAlmostEqual(overall_er[0], drift.metrics["avg_translation_error"])
        self.assertAlmostEqual(overall_er[1], drift.metrics["avg_rotation_error_radians"])

    def test_savefigures(self):
        drift = TranslationRotationDrift("test_dataset", "test_trajectory", "test_model", self.locations,
                                         self.tait_bryans, self.locations, self.tait_bryans,
                                         framerate=1, stepsize=1, align=False)
        with drift.figures_loader as figures:
            for key in figures.keys():
                figures[key].savefig("{}.png".format(key))


class TestCompoundTranslationRotationDrift(TestCase):
    def setUp(self) -> None:
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        self.tait_bryans = numpy.asarray(
            [[y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime]])

        self.locations = numpy.asarray([[0, 0, 0], [100, 0, 0], [200, 0, 0]])

    def test_Given_same_pose_and_poseGT_sequence_errors_should_be_0(self):
        drift1 = TranslationRotationDrift("test_dataset", "test_trajectory", "test_model", self.locations,
                                         self.tait_bryans, self.locations, self.tait_bryans,
                                         framerate=1, stepsize=1, align=False)

        drift2 = TranslationRotationDrift("test_dataset", "test_trajectory", "test_model", self.locations,
                                         self.tait_bryans, self.locations, self.tait_bryans,
                                         framerate=1, stepsize=1, align=False)

        compound_drift = CompoundTranslationRotationDrift("test_model", [drift1, drift2], 1)

        self.assertEqual(len(compound_drift.metrics["segment_errors"].keys()), 1)
        self.assertTrue(100 in compound_drift.metrics["segment_errors"].keys())
        self.assertEqual(len(compound_drift.metrics["segment_errors"][100]), 2)
        self.assertEqual(compound_drift.metrics["segment_errors"][100][0]["start_frame"], 0)
        self.assertAlmostEqual(compound_drift.metrics["segment_errors"][100][0]["rotation_error_percent"], 0)
        self.assertAlmostEqual(compound_drift.metrics["segment_errors"][100][0]["translation_error_percent"], 0)
        self.assertEqual(compound_drift.metrics["segment_errors"][100][0]["segment_lenght"], 100)
        self.assertAlmostEqual(compound_drift.metrics["segment_errors"][100][0]["speed"], 33.3333333)
        self.assertAlmostEqual(compound_drift.metrics["average_segment_errors"][100]["average_rotation_error"], 0)
        self.assertAlmostEqual(compound_drift.metrics["average_segment_errors"][100]["average_translation_error"], 0)
        self.assertEqual(len(compound_drift.metrics["average_segment_errors"][200]), 0)
        self.assertEqual(len(compound_drift.metrics["average_segment_errors"][300]), 0)
        self.assertEqual(len(compound_drift.metrics["average_segment_errors"][400]), 0)
        self.assertEqual(len(compound_drift.metrics["average_segment_errors"][500]), 0)
        self.assertEqual(len(compound_drift.metrics["average_segment_errors"][600]), 0)
        self.assertEqual(len(compound_drift.metrics["average_segment_errors"][700]), 0)
        self.assertEqual(len(compound_drift.metrics["average_segment_errors"][800]), 0)
        self.assertAlmostEqual(compound_drift.metrics["avg_translation_error"], 0)
        self.assertAlmostEqual(compound_drift.metrics["avg_rotation_error_radians"], 0)
        self.assertAlmostEqual(compound_drift.metrics["avg_translation_error_percent"], 0)
        self.assertAlmostEqual(compound_drift.metrics["avg_rotation_error_degrees_per_meter"], 0)

    def test_given_valid_poses_should_return_complete_metric(self):
        drift1 = TranslationRotationDrift("test_dataset", "test_trajectory", "test_model", self.locations,
                                         self.tait_bryans, self.locations, self.tait_bryans,
                                         framerate=1, stepsize=1, align=False)

        drift2 = TranslationRotationDrift("test_dataset", "test_trajectory", "test_model", self.locations,
                                         self.tait_bryans, self.locations, self.tait_bryans,
                                         framerate=1, stepsize=1, align=False)

        compound_drift = CompoundTranslationRotationDrift("test_model", [drift1, drift2], 1)

        self.assertEqual(compound_drift.metrics["dataset"], "'test_dataset'")
        self.assertTrue(compound_drift.metrics["trajectory"], "all")
        self.assertEqual(compound_drift.metrics["model"], "test_model")
        self.assertEqual(compound_drift.metrics["stepsize"], 1)
        self.assertTrue("segment_errors" in compound_drift.metrics.keys())
        self.assertTrue("average_segment_errors" in compound_drift.metrics.keys())
        self.assertTrue("avg_translation_error" in compound_drift.metrics.keys())
        self.assertTrue("avg_rotation_error_radians" in compound_drift.metrics.keys())
        self.assertTrue("avg_translation_error_percent" in compound_drift.metrics.keys())
        self.assertTrue("avg_rotation_error_degrees_per_meter" in compound_drift.metrics.keys())
        with compound_drift.figures_loader as figures:
            self.assertTrue("average_segment_error_plot" in figures.keys())

        self.assertEqual(compound_drift.metrics["avg_avg_translation_error"], 9.05808122401665e-18)
        self.assertAlmostEqual(compound_drift.metrics["avg_avg_rotation_error_radians"], 9.058081224016651e-16)
        self.assertEqual(compound_drift.metrics["avg_avg_translation_error_percent"], 9.058081224016651e-16)
        self.assertEqual(compound_drift.metrics["avg_avg_rotation_error_degrees_per_meter"], 0)

    def test_savefigures(self):
        drift1 = TranslationRotationDrift("test_dataset", "test_trajectory", "test_model", self.locations,
                                         self.tait_bryans, self.locations, self.tait_bryans,
                                         framerate=1, stepsize=1, align=False)

        drift2 = TranslationRotationDrift("test_dataset", "test_trajectory", "test_model", self.locations,
                                         self.tait_bryans, self.locations, self.tait_bryans,
                                         framerate=1, stepsize=1, align=False)

        compound_drift = CompoundTranslationRotationDrift("test_model", [drift1, drift2], 1)

        with compound_drift.figures_loader as figures:
            for key in figures.keys():
                figures[key].savefig("{}_compound.png".format(key))


class TestAbsoluteTrajectoryError(TestCase):
    def setUp(self) -> None:
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        self.tait_bryans = numpy.asarray(
            [[y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime]])

        self.locations = numpy.asarray([[0, 0, 0], [100, 0, 0], [200, 0, 0]])

    def test_Given_same_pose_and_poseGT_sequence_errors_should_be_0(self):
        ATE = AbsoluteTrajectoryError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)

        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_L2_norm'][0], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_L2_norm'][1], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_L2_norm'][2], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_stats']["rmse"], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_stats']["mean"], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_stats']["median"], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_stats']["std"], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_stats']["min"], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_stats']["max"], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_stats']["num_samples"], 3)
        numpy.testing.assert_almost_equal(ATE.metrics["absolute_trajectory_error"]['ATE_trans_vec'],
                                          numpy.zeros((3,3)))
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_degrees'][0], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_degrees'][1], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_degrees'][2], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_stats']["rmse"], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_stats']["mean"], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_stats']["median"], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_stats']["std"], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_stats']["min"], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_stats']["max"], 0)
        numpy.testing.assert_almost_equal(ATE.metrics["absolute_trajectory_error"]['ATE_rot_yaw_pitch_roll'],
                                          numpy.zeros((3, 3)))
        numpy.testing.assert_almost_equal(ATE.metrics["absolute_trajectory_error"]['scale_drift_percent'],
                                          [math.nan, 0, 0])
        self.assertTrue(math.isnan(ATE.metrics["absolute_trajectory_error"]['scale_drift_stats']["rmse"]))
        self.assertTrue(math.isnan(ATE.metrics["absolute_trajectory_error"]['scale_drift_stats']["mean"]))
        self.assertTrue(math.isnan(ATE.metrics["absolute_trajectory_error"]['scale_drift_stats']["median"]))
        self.assertTrue(math.isnan(ATE.metrics["absolute_trajectory_error"]['scale_drift_stats']["std"]))
        self.assertTrue(math.isnan(ATE.metrics["absolute_trajectory_error"]['scale_drift_stats']["min"]))
        self.assertTrue(math.isnan(ATE.metrics["absolute_trajectory_error"]['scale_drift_stats']["max"]))
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['scale_drift_stats']["num_samples"], 3)

    def test_given_kitti_07_poses_should_return_same_estimate_than_original_code_from_git(self):
        with open("kitti_07_location.pkl", "rb") as file:
            locations = pickle.load(file)
        with open("kitti_07_location_gt.pkl", "rb") as file:
            locations_gt = pickle.load(file)
        with open("kitti_07_orientation.pkl", "rb") as file:
            orientations = pickle.load(file)
        with open("kitti_07_orientation_gt.pkl", "rb") as file:
            orientations_gt = pickle.load(file)

        with open("rpg_abs_trajectory_errors.pkl", "rb") as file:
            expected_ATE = pickle.load(file)

        ATE = AbsoluteTrajectoryError("test_dataset", "test_trajectory", "test_model", locations_gt,
                                         orientations_gt, locations, orientations)

        actual_ATE = ATE.metrics["absolute_trajectory_error"]

        numpy.testing.assert_array_almost_equal(actual_ATE['ATE_trans_L2_norm'],
                                                expected_ATE['abs_e_trans'])
        self.assertEqual(actual_ATE['ATE_trans_stats']["rmse"], expected_ATE['abs_e_trans_stats']["rmse"])
        self.assertEqual(actual_ATE['ATE_trans_stats']["mean"], expected_ATE['abs_e_trans_stats']["mean"])
        self.assertEqual(actual_ATE['ATE_trans_stats']["median"], expected_ATE['abs_e_trans_stats']["median"])
        self.assertEqual(actual_ATE['ATE_trans_stats']["std"], expected_ATE['abs_e_trans_stats']["std"])
        self.assertEqual(actual_ATE['ATE_trans_stats']["min"], expected_ATE['abs_e_trans_stats']["min"])
        self.assertEqual(actual_ATE['ATE_trans_stats']["max"], expected_ATE['abs_e_trans_stats']["max"])
        self.assertEqual(actual_ATE['ATE_trans_stats']["num_samples"], expected_ATE['abs_e_trans_stats']["num_samples"])
        numpy.testing.assert_almost_equal(actual_ATE['ATE_trans_vec'],
                                          expected_ATE['abs_e_trans_vec'])
        numpy.testing.assert_array_almost_equal(actual_ATE['ATE_rot_degrees'], expected_ATE['abs_e_rot'])
        self.assertAlmostEqual(actual_ATE['ATE_rot_stats']["rmse"], expected_ATE['abs_e_rot_stats']["rmse"])
        self.assertAlmostEqual(actual_ATE['ATE_rot_stats']["mean"], expected_ATE['abs_e_rot_stats']["mean"])
        self.assertAlmostEqual(actual_ATE['ATE_rot_stats']["median"], expected_ATE['abs_e_rot_stats']["median"])
        self.assertAlmostEqual(actual_ATE['ATE_rot_stats']["std"], expected_ATE['abs_e_rot_stats']["std"])
        self.assertAlmostEqual(actual_ATE['ATE_rot_stats']["min"], expected_ATE['abs_e_rot_stats']["min"])
        self.assertAlmostEqual(actual_ATE['ATE_rot_stats']["max"], expected_ATE['abs_e_rot_stats']["max"])
        numpy.testing.assert_almost_equal(actual_ATE['ATE_rot_yaw_pitch_roll'],
                                          expected_ATE['abs_e_ypr'])
        numpy.testing.assert_almost_equal(actual_ATE['scale_drift_percent'],
                                          expected_ATE['abs_e_scale_perc'])
        self.assertAlmostEqual(actual_ATE['scale_drift_stats']["rmse"], expected_ATE['abs_e_scale_stats']["rmse"])
        self.assertAlmostEqual(actual_ATE['scale_drift_stats']["mean"], expected_ATE['abs_e_scale_stats']["mean"])
        self.assertAlmostEqual(actual_ATE['scale_drift_stats']["median"], expected_ATE['abs_e_scale_stats']["median"])
        self.assertTrue(math.isnan(actual_ATE['scale_drift_stats']["std"]))
        self.assertTrue(math.isnan(expected_ATE['abs_e_scale_stats']["std"]))
        self.assertAlmostEqual(actual_ATE['scale_drift_stats']["min"], expected_ATE['abs_e_scale_stats']["min"])
        self.assertAlmostEqual(actual_ATE['scale_drift_stats']["max"], expected_ATE['abs_e_scale_stats']["max"])
        self.assertEqual(actual_ATE['scale_drift_stats']["num_samples"], expected_ATE['abs_e_scale_stats']["num_samples"])

    def test_savefigures(self):
        ATE = AbsoluteTrajectoryError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)
        with ATE.figures_loader as figures:
            for key in figures.keys():
                figures[key].savefig("{}.png".format(key))


class TestCompoundAbsoluteTrajectoryError(TestCase):
    def setUp(self) -> None:
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        self.tait_bryans = numpy.asarray(
            [[y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime]])

        self.locations = numpy.asarray([[0, 0, 0], [100, 0, 0], [200, 0, 0]])

    def test_Given_same_pose_and_poseGT_sequence_errors_should_be_0(self):
        ATE1 = AbsoluteTrajectoryError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)

        ATE2 = AbsoluteTrajectoryError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)

        ATE = CompoundAbsoluteTrajectoryError("test_model", [ATE1, ATE2])

        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_L2_norm'][0], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_L2_norm'][1], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_L2_norm'][2], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_L2_norm'][3], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_L2_norm'][4], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_L2_norm'][5], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_stats']["rmse"], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_stats']["mean"], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_stats']["median"], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_stats']["std"], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_stats']["min"], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_stats']["max"], 0)
        self.assertEqual(ATE.metrics["absolute_trajectory_error"]['ATE_trans_stats']["num_samples"], 6)

        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_degrees'][0], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_degrees'][1], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_degrees'][2], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_degrees'][3], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_degrees'][4], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_degrees'][5], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_stats']["rmse"], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_stats']["mean"], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_stats']["median"], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_stats']["std"], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_stats']["min"], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_stats']["max"], 0)
        self.assertAlmostEqual(ATE.metrics["absolute_trajectory_error"]['ATE_rot_stats']["num_samples"], 6)

        self.assertAlmostEqual(ATE.metrics["avg_avg_ATE_trans_L2_norm"], 0)
        self.assertAlmostEqual(ATE.metrics["avg_avg_ATE_rot_degrees"], 0)

    def test_given_valid_poses_should_return_complete_metric(self):
        ATE1 = AbsoluteTrajectoryError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)

        ATE2 = AbsoluteTrajectoryError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)

        ATE = CompoundAbsoluteTrajectoryError("test_model", [ATE1, ATE2])

        self.assertEqual(ATE.metrics["dataset"], "'test_dataset'")
        self.assertTrue(ATE.metrics["trajectory"], "all")
        self.assertEqual(ATE.metrics["model"], "test_model")

    def test_savefigures(self):
        ATE1 = AbsoluteTrajectoryError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)

        ATE2 = AbsoluteTrajectoryError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)

        ATE = CompoundAbsoluteTrajectoryError("test_model", [ATE1, ATE2])

        with ATE.figures_loader as figures:
            for key in figures.keys():
                figures[key].savefig("{}_compound.png".format(key))


class TestRelativeError(TestCase):
    def setUp(self) -> None:
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        self.tait_bryans = numpy.asarray(
            [[y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime]])

        self.locations = numpy.asarray([[0, 0, 0], [11, 0, 0], [20, 0, 0], [100, 0, 0]])

    def test_Given_same_pose_and_poseGT_sequence_errors_should_be_0(self):
        RE = RelativeError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)

        numpy.testing.assert_almost_equal(RE.metrics["relative_error"][10]["rel_trans"], [0, 0])
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_stats"]["rmse"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_stats"]["mean"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_stats"]["median"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_stats"]["std"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_stats"]["min"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_stats"]["max"], 0)
        self.assertEqual(RE.metrics["relative_error"][10]["rel_trans_stats"]["num_samples"], 2)

        numpy.testing.assert_almost_equal(RE.metrics["relative_error"][10]["rel_trans_perc"], [0, 0])
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_perc_stats"]["mean"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_perc_stats"]["median"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_perc_stats"]["std"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_perc_stats"]["min"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_perc_stats"]["max"], 0)
        self.assertEqual(RE.metrics["relative_error"][10]["rel_trans_perc_stats"]["num_samples"], 2)

        numpy.testing.assert_almost_equal(RE.metrics["relative_error"][10]["rel_rot"], [0, 0])
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_stats"]["mean"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_stats"]["median"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_stats"]["std"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_stats"]["min"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_stats"]["max"], 0)
        self.assertEqual(RE.metrics["relative_error"][10]["rel_rot_stats"]["num_samples"], 2)

        numpy.testing.assert_almost_equal(RE.metrics["relative_error"][10]["rel_yaw"], [0, 0])
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_yaw_stats"]["mean"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_yaw_stats"]["median"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_yaw_stats"]["std"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_yaw_stats"]["min"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_yaw_stats"]["max"], 0)
        self.assertEqual(RE.metrics["relative_error"][10]["rel_yaw_stats"]["num_samples"], 2)

        numpy.testing.assert_almost_equal(RE.metrics["relative_error"][10]["rel_gravity"], [0, 0])
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_gravity_stats"]["mean"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_gravity_stats"]["median"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_gravity_stats"]["std"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_gravity_stats"]["min"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_gravity_stats"]["max"], 0)
        self.assertEqual(RE.metrics["relative_error"][10]["rel_gravity_stats"]["num_samples"], 2)

        numpy.testing.assert_almost_equal(RE.metrics["relative_error"][10]["rel_rot_deg_per_m"], [0, 0])
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_deg_per_m_stats"]["mean"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_deg_per_m_stats"]["median"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_deg_per_m_stats"]["std"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_deg_per_m_stats"]["min"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_deg_per_m_stats"]["max"], 0)
        self.assertEqual(RE.metrics["relative_error"][10]["rel_rot_deg_per_m_stats"]["num_samples"], 2)

        for i in range(20, 50, 10):
            numpy.testing.assert_almost_equal(RE.metrics["relative_error"][i]["rel_trans"], [])
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_stats"]["rmse"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_stats"]["mean"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_stats"]["median"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_stats"]["std"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_stats"]["min"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_stats"]["max"], 0)
            self.assertEqual(RE.metrics["relative_error"][i]["rel_trans_stats"]["num_samples"], 0)

            numpy.testing.assert_almost_equal(RE.metrics["relative_error"][i]["rel_trans_perc"], [])
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_perc_stats"]["mean"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_perc_stats"]["median"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_perc_stats"]["std"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_perc_stats"]["min"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_perc_stats"]["max"], 0)
            self.assertEqual(RE.metrics["relative_error"][i]["rel_trans_perc_stats"]["num_samples"], 0)

            numpy.testing.assert_almost_equal(RE.metrics["relative_error"][i]["rel_rot"], [])
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_stats"]["mean"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_stats"]["median"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_stats"]["std"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_stats"]["min"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_stats"]["max"], 0)
            self.assertEqual(RE.metrics["relative_error"][i]["rel_rot_stats"]["num_samples"], 0)

            numpy.testing.assert_almost_equal(RE.metrics["relative_error"][i]["rel_yaw"], [])
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_yaw_stats"]["mean"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_yaw_stats"]["median"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_yaw_stats"]["std"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_yaw_stats"]["min"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_yaw_stats"]["max"], 0)
            self.assertEqual(RE.metrics["relative_error"][i]["rel_yaw_stats"]["num_samples"], 0)

            numpy.testing.assert_almost_equal(RE.metrics["relative_error"][i]["rel_gravity"], [])
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_gravity_stats"]["mean"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_gravity_stats"]["median"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_gravity_stats"]["std"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_gravity_stats"]["min"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_gravity_stats"]["max"], 0)
            self.assertEqual(RE.metrics["relative_error"][i]["rel_gravity_stats"]["num_samples"], 0)

            numpy.testing.assert_almost_equal(RE.metrics["relative_error"][i]["rel_rot_deg_per_m"], [])
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_deg_per_m_stats"]["mean"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_deg_per_m_stats"]["median"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_deg_per_m_stats"]["std"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_deg_per_m_stats"]["min"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_deg_per_m_stats"]["max"], 0)
            self.assertEqual(RE.metrics["relative_error"][i]["rel_rot_deg_per_m_stats"]["num_samples"], 0)

    def test_given_kitti_07_poses_should_return_same_estimate_than_original_code_from_git(self):
        with open("kitti_07_location.pkl", "rb") as file:
            locations = pickle.load(file)
        with open("kitti_07_location_gt.pkl", "rb") as file:
            locations_gt = pickle.load(file)
        with open("kitti_07_orientation.pkl", "rb") as file:
            orientations = pickle.load(file)
        with open("kitti_07_orientation_gt.pkl", "rb") as file:
            orientations_gt = pickle.load(file)

        with open("rpg_relative_errors.pkl", "rb") as file:
            expected_RE = pickle.load(file)

        RE = RelativeError("test_dataset", "test_trajectory", "test_model", locations_gt,
                                         orientations_gt, locations, orientations)

        actual_RE = RE.metrics["relative_error"]

        for i in actual_RE.keys():
            numpy.testing.assert_almost_equal(actual_RE[i]["rel_trans"], expected_RE[i]["rel_trans"])
            self.assertAlmostEqual(actual_RE[i]["rel_trans_stats"]["rmse"], expected_RE[i]["rel_trans_stats"]["rmse"])
            self.assertAlmostEqual(actual_RE[i]["rel_trans_stats"]["mean"], expected_RE[i]["rel_trans_stats"]["mean"])
            self.assertAlmostEqual(actual_RE[i]["rel_trans_stats"]["median"], expected_RE[i]["rel_trans_stats"]["median"])
            self.assertAlmostEqual(actual_RE[i]["rel_trans_stats"]["std"], expected_RE[i]["rel_trans_stats"]["std"])
            self.assertAlmostEqual(actual_RE[i]["rel_trans_stats"]["min"], expected_RE[i]["rel_trans_stats"]["min"])
            self.assertAlmostEqual(actual_RE[i]["rel_trans_stats"]["max"], expected_RE[i]["rel_trans_stats"]["max"])
            self.assertEqual(actual_RE[i]["rel_trans_stats"]["num_samples"], expected_RE[i]["rel_trans_stats"]["num_samples"])

            numpy.testing.assert_almost_equal(actual_RE[i]["rel_trans_perc"], expected_RE[i]["rel_trans_perc"])
            self.assertAlmostEqual(actual_RE[i]["rel_trans_perc_stats"]["mean"], expected_RE[i]["rel_trans_perc_stats"]["mean"])
            self.assertAlmostEqual(actual_RE[i]["rel_trans_perc_stats"]["median"], expected_RE[i]["rel_trans_perc_stats"]["median"])
            self.assertAlmostEqual(actual_RE[i]["rel_trans_perc_stats"]["std"], expected_RE[i]["rel_trans_perc_stats"]["std"])
            self.assertAlmostEqual(actual_RE[i]["rel_trans_perc_stats"]["min"], expected_RE[i]["rel_trans_perc_stats"]["min"])
            self.assertAlmostEqual(actual_RE[i]["rel_trans_perc_stats"]["max"], expected_RE[i]["rel_trans_perc_stats"]["max"])
            self.assertEqual(actual_RE[i]["rel_trans_perc_stats"]["num_samples"], expected_RE[i]["rel_trans_perc_stats"]["num_samples"])

            numpy.testing.assert_almost_equal(actual_RE[i]["rel_rot"], expected_RE[i]["rel_rot"])
            self.assertAlmostEqual(actual_RE[i]["rel_rot_stats"]["mean"], expected_RE[i]["rel_rot_stats"]["mean"])
            self.assertAlmostEqual(actual_RE[i]["rel_rot_stats"]["median"], expected_RE[i]["rel_rot_stats"]["median"])
            self.assertAlmostEqual(actual_RE[i]["rel_rot_stats"]["std"], expected_RE[i]["rel_rot_stats"]["std"])
            self.assertAlmostEqual(actual_RE[i]["rel_rot_stats"]["min"], expected_RE[i]["rel_rot_stats"]["min"])
            self.assertAlmostEqual(actual_RE[i]["rel_rot_stats"]["max"], expected_RE[i]["rel_rot_stats"]["max"])
            self.assertEqual(actual_RE[i]["rel_rot_stats"]["num_samples"], expected_RE[i]["rel_rot_stats"]["num_samples"])

            numpy.testing.assert_almost_equal(actual_RE[i]["rel_yaw"], expected_RE[i]["rel_yaw"])
            self.assertAlmostEqual(actual_RE[i]["rel_yaw_stats"]["mean"], expected_RE[i]["rel_yaw_stats"]["mean"])
            self.assertAlmostEqual(actual_RE[i]["rel_yaw_stats"]["median"], expected_RE[i]["rel_yaw_stats"]["median"])
            self.assertAlmostEqual(actual_RE[i]["rel_yaw_stats"]["std"], expected_RE[i]["rel_yaw_stats"]["std"])
            self.assertAlmostEqual(actual_RE[i]["rel_yaw_stats"]["min"], expected_RE[i]["rel_yaw_stats"]["min"])
            self.assertAlmostEqual(actual_RE[i]["rel_yaw_stats"]["max"], expected_RE[i]["rel_yaw_stats"]["max"])
            self.assertEqual(actual_RE[i]["rel_yaw_stats"]["num_samples"], expected_RE[i]["rel_yaw_stats"]["num_samples"])

            numpy.testing.assert_almost_equal(actual_RE[i]["rel_gravity"], expected_RE[i]["rel_gravity"])
            self.assertAlmostEqual(actual_RE[i]["rel_gravity_stats"]["mean"], expected_RE[i]["rel_gravity_stats"]["mean"])
            self.assertAlmostEqual(actual_RE[i]["rel_gravity_stats"]["median"], expected_RE[i]["rel_gravity_stats"]["median"])
            self.assertAlmostEqual(actual_RE[i]["rel_gravity_stats"]["std"], expected_RE[i]["rel_gravity_stats"]["std"])
            self.assertAlmostEqual(actual_RE[i]["rel_gravity_stats"]["min"], expected_RE[i]["rel_gravity_stats"]["min"])
            self.assertAlmostEqual(actual_RE[i]["rel_gravity_stats"]["max"], expected_RE[i]["rel_gravity_stats"]["max"])
            self.assertEqual(actual_RE[i]["rel_gravity_stats"]["num_samples"], expected_RE[i]["rel_gravity_stats"]["num_samples"])

            numpy.testing.assert_almost_equal(actual_RE[i]["rel_rot_deg_per_m"], expected_RE[i]["rel_rot_deg_per_m"])
            self.assertAlmostEqual(actual_RE[i]["rel_rot_deg_per_m_stats"]["mean"], expected_RE[i]["rel_rot_deg_per_m_stats"]["mean"])
            self.assertAlmostEqual(actual_RE[i]["rel_rot_deg_per_m_stats"]["median"], expected_RE[i]["rel_rot_deg_per_m_stats"]["median"])
            self.assertAlmostEqual(actual_RE[i]["rel_rot_deg_per_m_stats"]["std"], expected_RE[i]["rel_rot_deg_per_m_stats"]["std"])
            self.assertAlmostEqual(actual_RE[i]["rel_rot_deg_per_m_stats"]["min"], expected_RE[i]["rel_rot_deg_per_m_stats"]["min"])
            self.assertAlmostEqual(actual_RE[i]["rel_rot_deg_per_m_stats"]["max"], expected_RE[i]["rel_rot_deg_per_m_stats"]["max"])
            self.assertEqual(actual_RE[i]["rel_rot_deg_per_m_stats"]["num_samples"], expected_RE[i]["rel_rot_deg_per_m_stats"]["num_samples"])


    def test_savefigures(self):
        RE = RelativeError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)

        with RE.figures_loader as figures:
            for key in figures.keys():
                figures[key].savefig("{}.png".format(key))


class TestCompoundRelativeError(TestCase):
    def setUp(self) -> None:
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        self.tait_bryans = numpy.asarray(
            [[y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime]])

        self.locations = numpy.asarray([[0, 0, 0], [11, 0, 0], [20, 0, 0], [100, 0, 0]])

    def test_Given_same_pose_and_poseGT_sequence_errors_should_be_0(self):
        RE1 = RelativeError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)
        RE2 = RelativeError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)

        RE = CompoundRelativeError("test_model", [RE1, RE2])

        numpy.testing.assert_almost_equal(RE.metrics["relative_error"][10]["rel_trans"], [0, 0, 0, 0])
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_stats"]["rmse"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_stats"]["mean"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_stats"]["median"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_stats"]["std"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_stats"]["min"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_stats"]["max"], 0)
        self.assertEqual(RE.metrics["relative_error"][10]["rel_trans_stats"]["num_samples"], 4)

        numpy.testing.assert_almost_equal(RE.metrics["relative_error"][10]["rel_trans_perc"], [0, 0, 0, 0])
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_perc_stats"]["mean"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_perc_stats"]["median"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_perc_stats"]["std"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_perc_stats"]["min"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_trans_perc_stats"]["max"], 0)
        self.assertEqual(RE.metrics["relative_error"][10]["rel_trans_perc_stats"]["num_samples"], 4)

        numpy.testing.assert_almost_equal(RE.metrics["relative_error"][10]["rel_rot"], [0, 0, 0, 0])
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_stats"]["mean"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_stats"]["median"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_stats"]["std"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_stats"]["min"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_stats"]["max"], 0)
        self.assertEqual(RE.metrics["relative_error"][10]["rel_rot_stats"]["num_samples"], 4)

        numpy.testing.assert_almost_equal(RE.metrics["relative_error"][10]["rel_yaw"], [0, 0, 0, 0])
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_yaw_stats"]["mean"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_yaw_stats"]["median"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_yaw_stats"]["std"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_yaw_stats"]["min"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_yaw_stats"]["max"], 0)
        self.assertEqual(RE.metrics["relative_error"][10]["rel_yaw_stats"]["num_samples"], 4)

        numpy.testing.assert_almost_equal(RE.metrics["relative_error"][10]["rel_gravity"], [0, 0, 0, 0])
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_gravity_stats"]["mean"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_gravity_stats"]["median"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_gravity_stats"]["std"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_gravity_stats"]["min"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_gravity_stats"]["max"], 0)
        self.assertEqual(RE.metrics["relative_error"][10]["rel_gravity_stats"]["num_samples"], 4)

        numpy.testing.assert_almost_equal(RE.metrics["relative_error"][10]["rel_rot_deg_per_m"], [0, 0, 0, 0])
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_deg_per_m_stats"]["mean"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_deg_per_m_stats"]["median"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_deg_per_m_stats"]["std"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_deg_per_m_stats"]["min"], 0)
        self.assertAlmostEqual(RE.metrics["relative_error"][10]["rel_rot_deg_per_m_stats"]["max"], 0)
        self.assertEqual(RE.metrics["relative_error"][10]["rel_rot_deg_per_m_stats"]["num_samples"], 4)

        for i in range(20, 50, 10):
            numpy.testing.assert_almost_equal(RE.metrics["relative_error"][i]["rel_trans"], [])
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_stats"]["rmse"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_stats"]["mean"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_stats"]["median"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_stats"]["std"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_stats"]["min"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_stats"]["max"], 0)
            self.assertEqual(RE.metrics["relative_error"][i]["rel_trans_stats"]["num_samples"], 0)

            numpy.testing.assert_almost_equal(RE.metrics["relative_error"][i]["rel_trans_perc"], [])
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_perc_stats"]["mean"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_perc_stats"]["median"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_perc_stats"]["std"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_perc_stats"]["min"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_trans_perc_stats"]["max"], 0)
            self.assertEqual(RE.metrics["relative_error"][i]["rel_trans_perc_stats"]["num_samples"], 0)

            numpy.testing.assert_almost_equal(RE.metrics["relative_error"][i]["rel_rot"], [])
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_stats"]["mean"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_stats"]["median"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_stats"]["std"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_stats"]["min"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_stats"]["max"], 0)
            self.assertEqual(RE.metrics["relative_error"][i]["rel_rot_stats"]["num_samples"], 0)

            numpy.testing.assert_almost_equal(RE.metrics["relative_error"][i]["rel_yaw"], [])
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_yaw_stats"]["mean"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_yaw_stats"]["median"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_yaw_stats"]["std"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_yaw_stats"]["min"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_yaw_stats"]["max"], 0)
            self.assertEqual(RE.metrics["relative_error"][i]["rel_yaw_stats"]["num_samples"], 0)

            numpy.testing.assert_almost_equal(RE.metrics["relative_error"][i]["rel_gravity"], [])
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_gravity_stats"]["mean"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_gravity_stats"]["median"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_gravity_stats"]["std"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_gravity_stats"]["min"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_gravity_stats"]["max"], 0)
            self.assertEqual(RE.metrics["relative_error"][i]["rel_gravity_stats"]["num_samples"], 0)

            numpy.testing.assert_almost_equal(RE.metrics["relative_error"][i]["rel_rot_deg_per_m"], [])
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_deg_per_m_stats"]["mean"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_deg_per_m_stats"]["median"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_deg_per_m_stats"]["std"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_deg_per_m_stats"]["min"], 0)
            self.assertAlmostEqual(RE.metrics["relative_error"][i]["rel_rot_deg_per_m_stats"]["max"], 0)
            self.assertEqual(RE.metrics["relative_error"][i]["rel_rot_deg_per_m_stats"]["num_samples"], 0)

    def test_given_valid_poses_should_return_complete_metric(self):
        RE1 = RelativeError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)
        RE2 = RelativeError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)

        RE = CompoundRelativeError("test_model", [RE1, RE2])

        self.assertEqual(RE.metrics["dataset"], "'test_dataset'")
        self.assertTrue(RE.metrics["trajectory"], "all")
        self.assertEqual(RE.metrics["model"], "test_model")

    def test_savefigures(self):
        RE1 = RelativeError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)
        RE2 = RelativeError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)

        RE = CompoundRelativeError("test_model", [RE1, RE2])

        with RE.figures_loader as figures:
            for key in figures.keys():
                figures[key].savefig("{}_compound.png".format(key))


class TestMetricLogger(TestCase):
    def setUp(self) -> None:
        y = 0.1745329
        x_prime = 0.3490659
        z_prime_prime = 0.7853982
        self.tait_bryans = numpy.asarray(
            [[y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime], [y, x_prime, z_prime_prime]])

        self.locations = numpy.asarray([[0, 0, 0], [11, 0, 0], [20, 0, 0], [100, 0, 0]])

        self.RE1 = RelativeError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)

        self.ATE1 = AbsoluteTrajectoryError("test_dataset", "test_trajectory", "test_model",
                                      self.locations,
                                      self.tait_bryans,
                                      self.locations,
                                      self.tait_bryans)

    def test_converts_all_arrays_to_list(self):
        logger = MetricLogger()
        # Return metrics contain no arrays
        self.assertFalse(self._contains_arrays(logger._arrays_to_lists(self.RE1.metrics)))
        # Original metrics preserved
        self.assertTrue(self._contains_arrays(self.RE1.metrics))

        # Return metrics contain no arrays
        self.assertFalse(self._contains_arrays(logger._arrays_to_lists(self.ATE1.metrics)))
        # Original metrics preserved
        self.assertTrue(self._contains_arrays(self.ATE1.metrics))

    def _contains_arrays(self, metrics: dict):
        contains_array = False
        for key in metrics.keys():
            if isinstance(metrics[key], dict):
                contains_array = contains_array or self._contains_arrays(metrics[key])
            elif isinstance(metrics[key], numpy.ndarray):
                contains_array = True
            else:
                continue
        return contains_array
