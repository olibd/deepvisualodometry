from unittest import TestCase

import numpy
import torch

from Models.Losses import LoopLoss


class LoopLossTest(TestCase):
    def setUp(self) -> None:
        segment = [[0,0,0], [1,1,1], [2,2,2], [10,10,10], [20, 20, 20], [12, 12, 12], [1, 1, 1]]
        batch = [segment, segment, segment]
        self.postions = torch.Tensor(batch)
        self.radius_threshold = 3
        self.loss = LoopLoss(loop_radius_threshold=self.radius_threshold)

    def test_compute_when_prediction_and_target_are_the_same_should_return_zero(self):
        loss = self.loss.compute(self.postions, self.postions)
        self.assertEqual(loss.numpy(), 0.)

    def test_compute_when_prediction_and_target_are_oposites_should_return_zero(self):
        loss = self.loss.compute(self.postions, -self.postions)
        self.assertEqual(loss.numpy(), 0.)

    def test_compute_loop_matrix(self):
        loops = self.loss._compute_loop_matrix(self.postions)

        for i in range(0,7):
            for j in range(0, 7):
                position1 = self.postions.numpy()[:, i, 0]
                position2 = self.postions.numpy()[:, j, 0]
                diff = position1 - position2
                expected_loops = numpy.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2) <= self.radius_threshold

                numpy.testing.assert_equal(loops.numpy()[:, i, j], expected_loops)

    def test_compute_location_distances(self):
        distances = self.loss._location_distances(self.postions)

        for i in range(0,7):
            for j in range(0, 7):
                position1 = self.postions.numpy()[:, i, 0]
                position2 = self.postions.numpy()[:, j, 0]
                diff = position1 - position2
                expected_distance = numpy.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
                numpy.testing.assert_equal(distances.numpy()[:, i, j], expected_distance)
