import signal
import traceback
from typing import Iterable, List

import numpy
import torch
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from stopit import SignalTimeout
from stopit.utils import BaseTimeout, TimeoutException

from Parameters import Parameters
from Logging import CometLogger


class TensorGeometry:
    @staticmethod
    def batch_assembleDeltaTranslationMatrices(relative_translation_batch: torch.Tensor) -> torch.Tensor:
        assert relative_translation_batch.shape[
                   2] is 3, "batch tensor should have shape of (batch_size, segment_length, 3)"
        device = relative_translation_batch.device
        # create an Nx1x3 zero tensor
        zeros = torch.zeros((relative_translation_batch.shape[0], 1, 3), requires_grad=relative_translation_batch.requires_grad).to(device)
        # add an extra 1x3 identity tensor indicating 0 rotation before the other rotations of the segments in the batch
        absolute_translation_batch = torch.cat([zeros, relative_translation_batch], 1)

        segment_positions = [absolute_translation_batch[:, 0]]
        for j in range(0, absolute_translation_batch.shape[1]-1):
            position = segment_positions[j] + absolute_translation_batch[:, j+1]
            segment_positions.append(position)

        segment_positions = torch.stack(segment_positions, dim=1)

        return segment_positions

    @staticmethod
    def batch_computeDeltaTranslations(global_positions_batch: torch.Tensor) -> torch.Tensor:

        # take the absolute translations tensor and make the translations relative
        return global_positions_batch[:, 1:] - global_positions_batch[:, 0:-1]

    @staticmethod
    def batch_assembleDeltaRotationMatrices(relative_rotation_batch: torch.Tensor) -> torch.Tensor:
        assert relative_rotation_batch.shape[2] is 3 and relative_rotation_batch.shape[3] is 3, \
            "batch tensor should have shape of (batch_size, segment_length, 3, 3)"

        # create an Nx1x3x3 identity tensor
        identity = torch.zeros((relative_rotation_batch.shape[0],1,3,3),
                               requires_grad=relative_rotation_batch.requires_grad) + torch.eye(3)
        device = relative_rotation_batch.device
        identity = identity.to(device)

        # add an extra 3x3 identity tensor indicating 0 rotation before the other rotations of the segments in the batch
        absolute_rotation_batch = torch.cat([identity, relative_rotation_batch], 1)

        segment_orientations = [absolute_rotation_batch[:, 0]]
        for j in range(0, absolute_rotation_batch.shape[1]-1):
            orientation = torch.matmul(segment_orientations[j], absolute_rotation_batch[:, j+1])
            segment_orientations.append(orientation)
        segment_orientations = torch.stack(segment_orientations, dim=1)

        return segment_orientations

    @staticmethod
    def batch_assembleDeltaEulerAngles(relative_rotation_batch: torch.Tensor) -> torch.Tensor:

        relative_rotation_batch = TensorGeometry.batchEulerAnglesToRotationMatrixTensor(relative_rotation_batch)

        absolute_rotation_batch = TensorGeometry.batch_assembleDeltaRotationMatrices(relative_rotation_batch)

        absolute_rotation_batch = TensorGeometry.batchRotationMatrixTensorToEulerAngles(absolute_rotation_batch)

        return absolute_rotation_batch

    @staticmethod
    def batchEulerAnglesToRotationMatrixTensor(batch: torch.Tensor) -> torch.Tensor:
        """
        Convert Tait-Bryant euler angles in radians to rotation a rotation matrix
        @param batched intrinsic y-x-z Tait-Bryant euler angles:
        @param x_prime:
        @param z_prime_prime:
        @return:
        """

        assert batch.shape[2] is 3, "batch tensor should have shape of (batch_size, segment_length, 3)"

        y = batch[:, :, 0]
        x_prime = batch[:, :, 1]
        z_prime_prime = batch[:, :, 2]

        device = batch.device

        assert y.requires_grad == x_prime.requires_grad and x_prime.requires_grad == z_prime_prime.requires_grad, \
            "All angle tensors must have the same requires_grad value"

        """
        [[1,        0,              0],
        [0, cos(x_prime), -sin(x_prime)],
        [0, sin(x_prime), cos(x_prime)]]
        """
        R_x = torch.zeros((x_prime.shape[0], x_prime.shape[1], 3, 3)).to(device)
        R_x[:, :, 0, 0] = 1
        R_x[:, :, 1, 1] = torch.cos(x_prime)
        R_x[:, :, 1, 2] = -torch.sin(x_prime)
        R_x[:, :, 2, 1] = torch.sin(x_prime)
        R_x[:, :, 2, 2] = torch.cos(x_prime)

        """
        [[cos(y), 0, sin(y)],
        [0      , 1,      0],
        [-sin(y), 0, cos(y)]]
        """
        R_y = torch.zeros((y.shape[0], y.shape[1], 3, 3)).to(device)
        R_y[:, :, 0, 0] = torch.cos(y)
        R_y[:, :, 0, 2] = torch.sin(y)
        R_y[:, :, 1, 1] = 1
        R_y[:, :, 2, 0] = -torch.sin(y)
        R_y[:, :, 2, 2] = torch.cos(y)

        """
        [[cos(z_prime_prime), -sin(z_prime_prime), 0],
        [sin(z_prime_prime),   cos(z_prime_prime), 0],
        [                   0,                  0, 1]]
        """
        R_z = torch.zeros((z_prime_prime.shape[0], z_prime_prime.shape[1], 3, 3)).to(device)
        R_z[:, :, 0, 0] = torch.cos(z_prime_prime)
        R_z[:, :, 0, 1] = -torch.sin(z_prime_prime)
        R_z[:, :, 1, 0] = torch.sin(z_prime_prime)
        R_z[:, :, 1, 1] = torch.cos(z_prime_prime)
        R_z[:, :, 2, 2] = 1

        R = torch.matmul(torch.matmul(R_y, R_x), R_z)
        return R

    @staticmethod
    def batchRotationMatrixTensorToEulerAngles(batch: torch.Tensor) -> torch.Tensor:
        # y-x-z Tait–Bryan angles intrinsic (ryxz)
        # the method code is modified from https://github.com/awesomebytes/delta_robot/blob/master/src/transformations.py

        i = 2
        j = 0
        k = 1
        repetition = 0
        frame = 1
        parity = 0

        # epsilon for testing whether a number is close to zero
        epsilon = numpy.finfo(float).eps * 4.0
        zero = torch.zeros((batch.shape[0], batch.shape[1])).to(batch.device).requires_grad_(batch.requires_grad)

        M = batch[:, :, :3, :3]
        if repetition:
            sy = torch.sqrt(M[:, :, i, j] * M[:, :, i, j] + M[:, :, i, k] * M[:, :, i, k])
            if sy > epsilon:
                ax = torch.atan2(M[:, :, i, j], M[:, :, i, k])
                ay = torch.atan2(sy, M[:, :, i, i])
                az = torch.atan2(M[:, :, j, i], -M[:, :, k, i])
            else:
                ax = torch.atan2(-M[:, :, j, k], M[:, :, j, j])
                ay = torch.atan2(sy, M[:, :, i, i])
                az = zero
        else:
            cy = torch.sqrt(M[:, :, i, i] * M[:, :, i, i] + M[:, :, j, i] * M[:, :, j, i])

            ax_true, ay_true, az_true, ax_false, ay_false, az_false = zero, zero, zero, zero, zero, zero

            # Matrix (batch_size, segment_size) that identifies which frames of a segments evaluates to True for the condition cy > epsilon
            # by multiplying this boolean matrix with M[:, :, i, j] we selects the proper rotation matrix values on which to evaluate atan2
            if (cy > epsilon).any():
                true_boolean_matrix = (cy > epsilon).type(torch.FloatTensor).to(batch.device)
                ax_true = torch.atan2(true_boolean_matrix * M[:, :, k, j], TensorGeometry._fix_negative_zero(true_boolean_matrix * M[:, :, k, k]))
                ay_true = torch.atan2(true_boolean_matrix * -M[:, :, k, i], cy)
                az_true = torch.atan2(true_boolean_matrix * M[:, :, j, i], TensorGeometry._fix_negative_zero(true_boolean_matrix * M[:, :, i, i]))

            # Matrix (batch_size, segment_size) that identifies which frames of a segments evaluated to False for the condition cy > epsilon
            # by multiplying this boolean matrix with M[:, :, i, j] we selects the proper rotation matrix values on which to evaluate atan2
            if (cy <= epsilon).any():
                false_boolean_matrix = (cy <= epsilon).type(torch.FloatTensor).to(batch.device)
                ax_false = torch.atan2(false_boolean_matrix * -M[:, :, j, k],
                                       TensorGeometry._fix_negative_zero(false_boolean_matrix * M[:, :, j, j]))
                ay_false = torch.atan2(false_boolean_matrix * -M[:, :, k, i], cy)
                az_false = zero

            ax = ax_true + ax_false
            ay = ay_true + ay_false
            az = az_true + az_false

        if parity:
            ax, ay, az = -ax, -ay, -az
        if frame:
            ax, az = az, ax
        return torch.stack((ax, ay, az), dim=2)

    @staticmethod
    def _fix_negative_zero(tensor: torch.Tensor):
        """
        This methods will convert any -0.0 elements to 0.0 so that atan2 outputs the proper value.
        @param tensor:
        @return:
        """
        ones = torch.ones(tensor.shape).to(tensor.device)
        ones[tensor == -0.0] = -1
        return ones * tensor

    @staticmethod
    def eulerAnglesToRotationMatrixTensor(y: torch.Tensor, x_prime: torch.Tensor, z_prime_prime: torch.Tensor) -> torch.Tensor:
        """
        Convert Tait-Bryant euler angles in radians to rotation a rotation matrix
        @param y:
        @param x_prime:
        @param z_prime_prime:
        @return:
        """
        assert y.requires_grad == x_prime.requires_grad and x_prime.requires_grad == z_prime_prime.requires_grad, \
            "All angle tensors must have the same requires_grad value"

        assert y.device == x_prime.device and x_prime.device == z_prime_prime.device, \
            "All angle tensors must be on the same device"

        device = y.device

        """
        [[1,        0,              0],
        [0, cos(x_prime), -sin(x_prime)],
        [0, sin(x_prime), cos(x_prime)]]
        """
        R_x = torch.zeros((3,3)).to(device)
        R_x[0, 0] = 1
        R_x[1, 1] = torch.cos(x_prime)
        R_x[1, 2] = -torch.sin(x_prime)
        R_x[2, 1] = torch.sin(x_prime)
        R_x[2, 2] = torch.cos(x_prime)

        """
        [[cos(y), 0, sin(y)],
        [0      , 1,      0],
        [-sin(y), 0, cos(y)]]
        """
        R_y = torch.zeros((3, 3)).to(device)
        R_y[0, 0] = torch.cos(y)
        R_y[0, 2] = torch.sin(y)
        R_y[1, 1] = 1
        R_y[2, 0] = -torch.sin(y)
        R_y[2, 2] = torch.cos(y)

        """
        [[cos(z_prime_prime), -sin(z_prime_prime), 0],
        [sin(z_prime_prime),   cos(z_prime_prime), 0],
        [                   0,                  0, 1]]
        """
        R_z = torch.zeros((3, 3)).to(device)
        R_z[0, 0] = torch.cos(z_prime_prime)
        R_z[0, 1] = -torch.sin(z_prime_prime)
        R_z[1, 0] = torch.sin(z_prime_prime)
        R_z[1, 1] = torch.cos(z_prime_prime)
        R_z[2, 2] = 1

        R = torch.mm(torch.mm(R_y, R_x), R_z)
        return R

    @staticmethod
    def rotation_matrix_to_euler_tait_bryan(matrix: torch.Tensor) -> torch.Tensor:
        # y-x-z Tait–Bryan angles intrinsic (ryxz)
        # the method code is taken from https://github.com/awesomebytes/delta_robot/blob/master/src/transformations.py

        i = 2
        j = 0
        k = 1
        repetition = 0
        frame = 1
        parity = 0

        # epsilon for testing whether a number is close to zero
        epsilon = numpy.finfo(float).eps * 4.0

        M = matrix[:3, :3]
        if repetition:
            sy = torch.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
            if sy > epsilon:
                ax = torch.atan2(M[i, j], M[i, k])
                ay = torch.atan2(sy, M[i, i])
                az = torch.atan2(M[j, i], -M[k, i])
            else:
                ax = torch.atan2(-M[j, k], M[j, j])
                ay = torch.atan2(sy, M[i, i])
                az = torch.Tensor([0.0]).to(matrix.device).requires_grad_(True).squeeze()
        else:
            cy = torch.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
            if cy > epsilon:
                ax = torch.atan2(M[k, j], M[k, k])
                ay = torch.atan2(-M[k, i], cy)
                az = torch.atan2(M[j, i], M[i, i])
            else:
                ax = torch.atan2(-M[j, k], M[j, j])
                ay = torch.atan2(-M[k, i], cy)
                az = torch.Tensor([0.0]).to(matrix.device).requires_grad_(True).squeeze()

        if parity:
            ax, ay, az = -ax, -ay, -az
        if frame:
            ax, az = az, ax
        return torch.cat((ax.unsqueeze(0), ay.unsqueeze(0), az.unsqueeze(0))).squeeze(0)

    @staticmethod
    def batch_eulerDifferences(absolute_orientation_batch: torch.Tensor) -> torch.Tensor:
        absolute_orientation_batch = TensorGeometry.batchEulerAnglesToRotationMatrixTensor(absolute_orientation_batch)
        matrix_diff = torch.matmul(absolute_orientation_batch[:, 1:], absolute_orientation_batch[:, 0:-1].inverse())
        return TensorGeometry.batchRotationMatrixTensorToEulerAngles(matrix_diff)

class Geometry:
    @staticmethod
    def matrix_to_quaternion(rotation_matrix: numpy.ndarray):
        rotation_matrix = rotation_matrix.reshape((3, 3))
        try:
            initial_orientation_quat = Quaternion(matrix=rotation_matrix).unit
        except ValueError:
            # Using Scipy rotation library to approximate an orthogonal matrix as described in:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_matrix.html
            initial_orientation_quat = Quaternion(
                matrix=Rotation.from_matrix(rotation_matrix).as_matrix()).unit

        return initial_orientation_quat

    @staticmethod
    def matrix_to_tait_bryan_euler(rotation_matrix: numpy.ndarray):
        return Geometry.quaternion_to_tait_bryan_rotation(Geometry.matrix_to_quaternion(rotation_matrix))

    @staticmethod
    def quaternion_to_matrix(quaternion: Quaternion) -> numpy.ndarray:
        rotation_matrix = quaternion.rotation_matrix
        rotation_matrix = rotation_matrix.reshape((9,))
        return rotation_matrix

    @staticmethod
    def remap_position_axes(rotation: Quaternion, positions: numpy.ndarray) -> numpy.ndarray:
        """
        Rotate the position vector according to the rotation given
        @param rotation:
        @param positions:
        @return:
        """
        rotation_matrix: numpy.ndarray = rotation.rotation_matrix
        remapped_positions: numpy.ndarray = positions.copy()
        for position in remapped_positions:
            position[:] = torch.FloatTensor(rotation_matrix.dot(position))

        return remapped_positions

    @staticmethod
    def assemble_delta_translations(delta_translation: numpy.ndarray) -> numpy.ndarray:
        """
        Assembles an array of translation deltas into a complete sequence of positions
        @param delta_translation: Array of translation deltas
        @return: Array describing a sequence of positions
        """
        positions = numpy.zeros((delta_translation.shape[0] + 1, 3))
        for i, translation in enumerate(delta_translation):
            positions[i + 1] = positions[i] + translation

        return positions

    @staticmethod
    def assemble_batch_delta_translations(batch_delta_translation: numpy.ndarray) -> numpy.ndarray:
        assert len(batch_delta_translation.shape) == 3
        assembled_segments = []
        for segment in batch_delta_translation:
            assembled_segments.append(Geometry.assemble_delta_translations(segment))

        return numpy.asarray(assembled_segments)

    @staticmethod
    def reset_positions_to_origin(initial_position: numpy.ndarray, positions: numpy.ndarray) -> numpy.ndarray:
        """Reset the sequence's translation relative to the first frame
        @param initial_position: x-y-z array representing the initial position
        @param positions: Array of positions
        @return a list of PyQuaternion objects
        """
        resetted_positions = positions.copy()
        for position in resetted_positions:
            position[:] = position + initial_position

        return resetted_positions

    @staticmethod
    def assemble_delta_quaternion_rotations(delta_rotations: list) -> List[Quaternion]:
        """
        Assembles a list of rotation deltas into a complete sequence of orientations
        @param delta_rotations: List of PyQuaternion objects describing rotation deltas
        @return: list of PyQuaternion objects describing a sequence of orientations
        """
        orientations = [Quaternion()]
        for i, rotation in enumerate(delta_rotations):
            orientations.append(rotation * orientations[i])

        return orientations

    @staticmethod
    def assemble_delta_tait_bryan_rotations(delta_rotations: numpy.ndarray) -> numpy.ndarray:
        """
        Assembles an array of rotation deltas into a complete sequence of orientations
        @param delta_rotations: Array of tait-bryan angles describing rotation deltas
        @return: Array of tait-bryan angles describing a sequence of orientations
        """
        orientations = numpy.asarray([[0.0, 0.0, 0.0]])
        for i, rotation in enumerate(delta_rotations):
            rotation_quat = Geometry.tait_bryan_rotation_to_quaternion(rotation)
            orientation_quat = Geometry.tait_bryan_rotation_to_quaternion(orientations[i])
            orientation = Geometry.quaternion_to_tait_bryan_rotation(rotation_quat * orientation_quat)
            orientation = numpy.asarray([orientation])
            orientations = numpy.concatenate((orientations, orientation), axis=0)

        return orientations

    @staticmethod
    def assemble_batch_delta_tait_bryan_rotations(batch_delta_rotation: numpy.ndarray) -> numpy.ndarray:
        assert len(batch_delta_rotation.shape) == 3
        assembled_segments = []
        for segment in batch_delta_rotation:
            assembled_segments.append(Geometry.assemble_delta_tait_bryan_rotations(segment))

        return numpy.asarray(assembled_segments)

    @staticmethod
    def reset_orientations_to_origin(initial_orientation_quat: Quaternion, orientations: list) -> List[Quaternion]:
        """Reset the sequence's orientations relative to the first frame. Expects angles to be in quaternions
        @param initial_orientation_quat: Quaternion representing the initial orientation
        @param orientations: list of Quaternions representing current orientations
        @return a list of PyQuaternion objects representing a sequence of orientations relative to the initial one
        """
        resetted_orientations = []

        for i, orientation in enumerate(orientations):
            current_orientation_quat = Quaternion(orientation)
            resetted_oriention_quat = initial_orientation_quat * current_orientation_quat
            resetted_orientations.append(resetted_oriention_quat)

        return resetted_orientations

    @staticmethod
    def reset_euler_orientations_to_origin(initial_orientation_euler: numpy.ndarray, orientations: numpy.ndarray) -> \
            numpy.ndarray:
        """Reset the sequence's orientations relative to the first frame. Expects angles to be in radians
        @param initial_orientation: Tait-bryan angle representing the initial orientation
        @param orientations: array of tait-bryan representing current orientations
        @return array of tait-bryan angles representing a sequence of orientations relative to the initial one
        """

        initial_orientation_quat = Geometry.tait_bryan_rotation_to_quaternion(initial_orientation_euler)
        orientations_quat = Geometry.tait_bryan_rotations_to_quaternions(orientations)

        resetted_orientations = Geometry.reset_orientations_to_origin(initial_orientation_quat, orientations_quat)

        return numpy.asarray(Geometry.quaternions_to_tait_bryan_rotations(resetted_orientations))

    @staticmethod
    def rotation_matrices_to_quaternions(rotation_matrices: Iterable) -> List[Quaternion]:
        """
        Type casts an iterable collection of flat rotation matrices to a list of PyQuaternion objects
        @param rotation_matrices: Iterable collection of attitudes
        @return a list of PyQuaternion objects
        """
        return [Geometry.matrix_to_quaternion(rot_matrix) for rot_matrix in rotation_matrices]

    @staticmethod
    def rotation_matrices_to_euler(rotation_matrices: Iterable) -> numpy.ndarray:
        """
        Type casts an iterable collection of flat rotation matrices to an array of tait-bryan angles
        @param rotation_matrices: Iterable collection of attitudes
        @return array of tait-bryan angles
        """
        return numpy.asarray([Geometry.matrix_to_tait_bryan_euler(rot_matrix) for rot_matrix in rotation_matrices])

    @staticmethod
    def quaternion_elements_to_quaternions(quaternion_elements: Iterable) -> List[Quaternion]:
        """
        Type casts an iterable collection of quaternion elements (w,x,y,z) to a list of PyQuaternion objects
        @param quaternion_elements: Iterable collection of quaternion elements (w,x,y,z)
        @return a list of PyQuaternion objects
        """
        return [Quaternion(elements) for elements in quaternion_elements]

    @staticmethod
    def tait_bryan_rotations_to_quaternions(rotations: Iterable) -> List[Quaternion]:
        """
        Type casts an iterable collection of intrinsic y-x'-z'' tait-bryan rotations to a list of PyQuaternion objects
        @param rotations: Iterable collection of intrinsic y-x'-z'' tait-bryan rotations
        @return a list of PyQuaternion objects
        """
        return [Geometry.tait_bryan_rotation_to_quaternion(rotation) for rotation in rotations]

    @staticmethod
    def tait_bryan_rotation_to_quaternion(rotation: numpy.ndarray) -> Quaternion:
        """
        Type casts an intrinsic y-x'-z'' tait-bryan rotation to a PyQuaternion objects
        @param rotation: intrinsic y-x'-z'' tait-bryan rotation
        @return PyQuaternion objects
        """
        # att.elements[[3, 0, 1, 2]] reorganizes quaternion elements from scalar last x-y-z-w to scalar first w-x-y-z
        return Quaternion(Rotation.from_euler("YXZ", rotation).as_quat()[[3, 0, 1, 2]])

    @staticmethod
    def tait_bryan_rotation_to_matrix(rotation: numpy.ndarray) -> numpy.ndarray:
        """
        Type casts an intrinsic y-x'-z'' tait-bryan rotation to a Rotation matrix
        @param rotation: intrinsic y-x'-z'' tait-bryan rotation
        @return Rotation matrix
        """
        return Rotation.from_euler("YXZ", rotation).as_matrix()

    @staticmethod
    def tait_bryan_rotations_to_matrices(rotations: numpy.ndarray) -> numpy.ndarray:
        """
        Type casts a collection of intrinsic y-x'-z'' tait-bryan rotations to a collection of Rotation matrix
        @param rotations: intrinsic y-x'-z'' tait-bryan rotation
        @return Rotations matrices
        """
        return Geometry.tait_bryan_rotation_to_matrix(rotations)

    @staticmethod
    def poses_to_transformations_matrix(locations: numpy.ndarray, tait_bryan_orientations: numpy.ndarray) -> numpy.ndarray:
        """
        Converts a collection of locations and intrinsic y-x'-z'' tait-bryan orientations to a collection of transformation matrix
        @param locations: x,y,z locations
        @param tait_bryan_orientations: intrinsic y-x'-z'' tait-bryan orientations
        @return Transformation matrices
        """
        orientation_matrices = Geometry.tait_bryan_rotations_to_matrices(tait_bryan_orientations)
        transformations = numpy.zeros((locations.shape[0], 4, 4))
        transformations[:, 0, 3] = locations[:, 0]
        transformations[:, 1, 3] = locations[:, 1]
        transformations[:, 2, 3] = locations[:, 2]
        transformations[:, 3, 3] = 1

        transformations[:, :3, :3] = orientation_matrices
        return transformations

    @staticmethod
    def quaternions_to_tait_bryan_rotations(quaternions: list) -> List[numpy.ndarray]:
        return [Geometry.quaternion_to_tait_bryan_rotation(quat)
                for quat in quaternions]

    @staticmethod
    def quaternion_to_tait_bryan_rotation(quat: Quaternion) -> numpy.ndarray:
        # att.elements[[1, 2, 3, 0]] reorganizes quaternion elements from scalar first w-x-y-z to scalar last x-y-z-w
        # We do this because PyQuaternion uses scalar first but Scipy.Rotation uses scalar last
        # output is intrinsic Tait-Bryan angles following the y-x'-z''
        return Rotation.from_quat(quat.elements[[1, 2, 3, 0]]).as_euler("YXZ")

    @staticmethod
    def get_position_differences(positions: numpy.ndarray) -> numpy.ndarray:
        translations = positions[1:] - positions[0:-1]
        return translations

    @staticmethod
    def get_tait_bryan_orientation_differences(orientations: numpy.ndarray) -> numpy.ndarray:
        orientations = Geometry.tait_bryan_rotations_to_quaternions(orientations)
        if not all(isinstance(att, Quaternion) for att in orientations):
            raise TypeError('Not all objects are of type Quaternion')

        rotations = []
        last_frame_rotation = None
        for orientation in orientations:
            if last_frame_rotation is not None:
                current_orientation_quat = orientation
                att_diff = current_orientation_quat * last_frame_rotation.inverse
                last_frame_rotation = current_orientation_quat
                rotations.append(att_diff)
            else:
                last_frame_rotation = orientation
        return numpy.asarray(Geometry.quaternions_to_tait_bryan_rotations(rotations))


def cuda_is_available():
    return torch.cuda.is_available() and not Parameters.force_run_on_cpu


class TracebackSignalTimeout(SignalTimeout):
    """Context manager for limiting in the time the execution of a block
    using signal.SIGALRM Unix signal.

    See :class:`stopit.utils.BaseTimeout` for more information
    """
    def __init__(self, seconds, swallow_exc=True):
        seconds = int(seconds)  # alarm delay for signal MUST be int
        self.traceback = None
        super(SignalTimeout, self).__init__(seconds, swallow_exc)

    def handle_timeout(self, signum, frame):
        self.state = BaseTimeout.TIMED_OUT
        self.traceback = traceback.format_stack(frame)
        d = {'_frame': frame}  # Allow access to frame object.
        d.update(frame.f_globals)  # Unless shadowed by global
        d.update(frame.f_locals)

        message = "Timeout Signal received.\nTraceback:\n"
        message += ''.join(self.traceback)

        CometLogger.print(message)

        exception_message = 'Block exceeded maximum timeout value (%d seconds). \nTraceback:' % self.seconds
        exception_message += ''.join(self.traceback)
        raise TimeoutException(exception_message)