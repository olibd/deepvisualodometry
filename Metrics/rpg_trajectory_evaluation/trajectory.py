import numpy as np
from .trajectory_utils import get_distance_from_start
from .results_writer import compute_statistics
from .compute_trajectory_errors import compute_absolute_error, compute_relative_error
from .align_utils import alignTrajectory
from .metrics import kRelMetrics

from .transformations import quaternion_from_matrix, quaternion_matrix

"""
Code modified from https://github.com/uzh-rpg/rpg_trajectory_evaluation
"""
class Trajectory:
    default_boxplot_perc = [0.1, 0.2, 0.3, 0.4, 0.5]

    def __init__(self, location: np.ndarray, location_gt: np.ndarray, orientation: np.ndarray,
                 orientation_gt: np.ndarray, platform='', alg_name='', dataset_name='',
                 align_type='sim3', align_num_frames=-1,
                 est_type='traj_est',
                 preset_boxplot_distances=[],
                 preset_boxplot_percentages=[]):

        assert align_type in ['first_frame', 'sim3', 'se3']
        self.data_loaded = self._load_data(location, location_gt, orientation, orientation_gt)

        # information of the results, useful as labels
        self.platform = platform
        self.alg = alg_name
        self.dataset_short_name = dataset_name
        self.uid = self.platform + '_' + self.alg + '_' +\
            self.dataset_short_name
        self.est_type = est_type
        self.success = False

        self.data_loaded = False
        self.data_aligned = False

        self.align_type = align_type
        self.align_num_frames = int(align_num_frames)

        self.align_str = self.align_type + '_' + str(self.align_num_frames)

        self.start_time_sec = -float('inf')
        self.end_time_sec = float('inf')

        self.abs_errors = {}

        # we cache relative error since it is time-comsuming to compute
        self.rel_errors = {}

        self.boxplot_pcts = preset_boxplot_percentages
        if len(preset_boxplot_distances) != 0:
            self.preset_boxplot_distances = preset_boxplot_distances
        else:
            if not self.boxplot_pcts:
                self.boxplot_pcts = Trajectory.default_boxplot_perc
            self._compute_boxplot_distances()

        self._align_trajectory()

    def _load_data(self, location, location_gt, orientation, orientation_gt):
        """
        Loads the trajectory data. The resuls {p_es, q_es, p_gt, q_gt} is
        synchronized and has the same length.
        """
        self.t_es = list(range(len(location)))
        self.p_es = location
        self.q_es = orientation
        self.t_gt = list(range(len(location_gt)))
        self.p_gt = location_gt
        self.q_gt = orientation_gt
        self.t_gt_raw = self.t_gt
        self.p_gt_raw = self.p_gt
        self.q_gt_raw = self.q_gt

        self.accum_distances = get_distance_from_start(self.p_gt_raw)
        self.traj_length = self.accum_distances[-1]
        self.accum_distances = get_distance_from_start(self.p_gt)

        return True

    def _compute_boxplot_distances(self):
        self.preset_boxplot_distances = [np.floor(pct*self.traj_length)
                                         for pct in self.boxplot_pcts]

    def _align_trajectory(self):
        if self.data_aligned:
            return

        n = int(self.align_num_frames)
        if n < 0.0:
            n = len(self.p_es)

        self.trans = np.zeros((3,))
        self.rot = np.eye(3)
        self.scale = 1.0
        if self.align_type == 'none':
            pass
        else:
            self.scale, self.rot, self.trans = alignTrajectory(
                self.p_es, self.p_gt, self.q_es, self.q_gt,
                self.align_type, self.align_num_frames)

        self.p_es_aligned = np.zeros(np.shape(self.p_es))
        self.q_es_aligned = np.zeros(np.shape(self.q_es))
        for i in range(np.shape(self.p_es)[0]):
            self.p_es_aligned[i, :] = self.scale * \
                self.rot.dot(self.p_es[i, :]) + self.trans
            q_es_R = self.rot.dot(
                quaternion_matrix(self.q_es[i, :])[0:3, 0:3])
            q_es_T = np.identity(4)
            q_es_T[0:3, 0:3] = q_es_R
            self.q_es_aligned[i, :] = quaternion_from_matrix(q_es_T)

        self.data_aligned = True

    def compute_absolute_error(self) -> dict:
        if self.abs_errors:
            pass
        else:
            # align trajectory if necessary
            self._align_trajectory()
            e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc =\
                compute_absolute_error(self.p_es_aligned,
                                       self.q_es_aligned,
                                       self.p_gt,
                                       self.q_gt)
            stats_trans = compute_statistics(e_trans)
            stats_rot = compute_statistics(e_rot)
            stats_scale = compute_statistics(e_scale_perc)

            self.abs_errors['ATE_trans_L2_norm'] = e_trans
            self.abs_errors['ATE_trans_stats'] = stats_trans

            self.abs_errors['ATE_trans_vec'] = e_trans_vec

            self.abs_errors['ATE_rot_degrees'] = e_rot
            self.abs_errors['ATE_rot_stats'] = stats_rot

            self.abs_errors['ATE_rot_yaw_pitch_roll'] = e_ypr

            self.abs_errors['scale_drift_percent'] = e_scale_perc
            self.abs_errors['scale_drift_stats'] = stats_scale
        return self.abs_errors

    def _compute_relative_error_at_subtraj_len(self, subtraj_len,
                                               max_dist_diff=-1):
        if max_dist_diff < 0:
            max_dist_diff = 0.2 * subtraj_len

        if self.rel_errors and (subtraj_len in self.rel_errors):
            pass
        else:
            Tcm = np.identity(4)
            _, e_trans, e_trans_perc, e_yaw, e_gravity, e_rot, e_rot_deg_per_m =\
                compute_relative_error(
                    self.p_es, self.q_es, self.p_gt, self.q_gt, Tcm,
                    subtraj_len, max_dist_diff, self.accum_distances,
                    self.scale)
            dist_rel_err = {'rel_trans': e_trans,
                            'rel_trans_stats':
                            compute_statistics(e_trans),
                            'rel_trans_perc': e_trans_perc,
                            'rel_trans_perc_stats':
                            compute_statistics(e_trans_perc),
                            'rel_rot': e_rot,
                            'rel_rot_stats':
                            compute_statistics(e_rot),
                            'rel_yaw': e_yaw,
                            'rel_yaw_stats':
                            compute_statistics(e_yaw),
                            'rel_gravity': e_gravity,
                            'rel_gravity_stats':
                            compute_statistics(e_gravity),
                            'rel_rot_deg_per_m': e_rot_deg_per_m,
                            'rel_rot_deg_per_m_stats':
                            compute_statistics(e_rot_deg_per_m)}
            self.rel_errors[subtraj_len] = dist_rel_err
        return True

    def compute_relative_errors(self, subtraj_lengths=[]) -> dict:
        suc = True
        if subtraj_lengths:
            for l in subtraj_lengths:
                suc = suc and self._compute_relative_error_at_subtraj_len(l)
        else:
            for l in self.preset_boxplot_distances:
                suc = suc and self._compute_relative_error_at_subtraj_len(l)
        self.success = suc

        return self.rel_errors

    def get_relative_errors_and_distances(
            self, error_types=['rel_trans', 'rel_trans_perc', 'rel_yaw']):
        rel_errors = {}
        for err_i in error_types:
            assert err_i in kRelMetrics
            rel_errors[err_i] = [[self.rel_errors[d][err_i]
                                 for d in self.preset_boxplot_distances]]
        return rel_errors, self.preset_boxplot_distances
