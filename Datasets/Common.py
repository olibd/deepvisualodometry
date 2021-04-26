import random

from torch.utils.data import BatchSampler

from .Interfaces import AbstractSegmentDataset


class SortedRandomBatchSegmentSampler(BatchSampler):
    def __init__(self, dataset: AbstractSegmentDataset, batch_size: int, drop_last: bool = False):
        self.dataset: AbstractSegmentDataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.unique_segment_len: dict = self._compute_unique_segment_len()
        self.length: int = self._compute_number_of_batches()

    def _compute_number_of_batches(self) -> int:
        total_nbr_batches = 0

        for _, segment_indices in self.unique_segment_len.items():
            n_batch = int(len(segment_indices) / self.batch_size)
            if not self.drop_last and len(segment_indices) % self.batch_size != 0:
                n_batch += 1
            total_nbr_batches += n_batch
        return total_nbr_batches

    def _compute_unique_segment_len(self) -> dict:
        unique_seq_lengths: dict = dict()
        for i, segment in enumerate(self.dataset.get_segments()):
            seg_len = len(segment)
            if seg_len in unique_seq_lengths:
                unique_seq_lengths[seg_len].append(i)
            else:
                unique_seq_lengths[seg_len] = [i]

        return unique_seq_lengths

    def __iter__(self):

        list_batch_indexes = []
        for _, segment_indices in self.unique_segment_len.items():
            nbr_segments = len(segment_indices)

            # re-arrange segments order randomly
            random.shuffle(segment_indices)
            # split indices to create equal size batch of indices
            tmp = [segment_indices[i:i+self.batch_size] for i in range(0, nbr_segments, self.batch_size)]

            if self.drop_last and len(tmp[-1]) < self.batch_size:
                tmp = tmp[:-1]

            list_batch_indexes += tmp

        return iter(list_batch_indexes)

    def __len__(self):
        return self.length
