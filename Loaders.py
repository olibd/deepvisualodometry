from torch import optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from Common.Helpers import cuda_is_available
from Datasets import KITTI, MidAir
from Datasets.Common import SortedRandomBatchSegmentSampler
from Datasets.KITTI import KITTIDataSegmenter, KITTIImageSequenceDatasetEulerDifferences
from Datasets.MidAir import MidAirDataSegmenter, MidAirImageSequenceDatasetEulerDifferences
from Logging import CometLogger
from Models import Losses
from Models.CoordConvDeepVO import CoordConvDeepVO
from Models.CoordConvSelfAttentionVO import CoordConvSelfAttentionVO
from Models.DeepVO import DeepVO
from Models.GlobalRelativeSelfAttentionVO import GlobalRelativeSelfAttentionVO
from Models.GlobalRelativeSelfAttentionVO_globXasKeyVal import GlobalRelativeSelfAttentionVO_globXasKeyVal
from Models.GlobalRelativeTransformerVO import GlobalRelativeTransformerVO
from Models.GlobalRelativeTransformerVO_globXAsKeyVal import GlobalRelativeTransformerVO_globXAsKeyVal
from Models.Losses import *
from Models.MagicVO import MagicVO
from Models.NoSelfAttentionVO import NoSelfAttentionVO
from Models.PositionalSimpleSelfAttentionVO import PositionalSimpleSelfAttentionVO
from Models.SelfAttentionVO import SelfAttentionVO
from Models.SelfAttentionVO_GlobRelOutput import SelfAttentionVO_GlobRelOutput
from Models.SimpleSelfAttentionVO import SimpleSelfAttentionVO
from Models.SkippedSelfAttention import SkippedSelfAttention
from Models.SnailSelfAttentionVO import SnailSelfAttentionVO
from Models.SnailVO import SnailVO
from Models.SplitSelfAttentionVO import SplitSelfAttentionVO
from Models.StackedSelfAttentionVO import StackedSelfAttentionVO
from Models.WeightedSelfAttentionVO import WeightedSelfAttentionVO
from Parameters import Parameters


def load_dataset_dataloaders(param: Parameters) -> tuple:
    if param.dataset == "KITTI":
        CometLogger.print("Using dataset source: KITTI")
        train_dataset, train_dataloader, valid_dataset, valid_dataloader = _load_kitti(param)
    elif param.dataset == "MidAir":
        CometLogger.print("Using dataset source: MidAir")
        train_dataset, train_dataloader, valid_dataset, valid_dataloader = _load_midAir_dataset(param)
    elif param.dataset == "all":
        CometLogger.print("Using dataset source: All")
        train_dataset, train_dataloader, valid_dataset, valid_dataloader = _load_all_datasets(param)
    else:
        raise NotImplementedError()

    return train_dataloader, valid_dataloader


def load_all_test_trajectory_dataloaders(param: Parameters) -> list:
    """
    Will load each test trajectory from each dataset defined in the parameters into its own dataloader.
    The trajectory will be segmented into multiple segments of size param.sliding_window_size, with
    an overlap of param.sliding_window_overlap.
    @param param:
    @return: A list of tuples (dataset name, video number, dataloader). One for each test trajectory
    """
    dataloaders = _load_kitti_test_videos_dataloaders(param)
    dataloaders.extend(_load_midair_test_trajectories_dataloaders(param))

    return dataloaders


def _load_kitti_test_videos_dataloaders(param: Parameters) -> list:
    """
    Will load each KITTI test videos defined in the parameters into its own dataloader.
    The trajectory will be segmented into multiple segments of size param.sliding_window_size, with
    an overlap of param.sliding_window_overlap.
    @param param:
    @return: A list of tuples ("KITTI", video number, dataloader). One for each test video
    """
    dataloaders = []

    for video in param.kitti_test_video:
        segment_destination = param.kitti_test_segments_directory+"/{}.pickle".format(video)
        kitti_test_dataset = KITTIImageSequenceDatasetEulerDifferences(segment_destination,
                                                                       new_size=(param.img_w, param.img_h),
                                                                       img_mean=param.kitti_mean,
                                                                       img_std=param.kitti_std,
                                                                       resize_mode=param.resize_mode,
                                                                       minus_point_5=param.minus_point_5,
                                                                       augment_dataset=param.test_dataset_augmentation)
        dataloader = DataLoader(kitti_test_dataset, num_workers=0,
                              pin_memory=param.pin_mem, batch_size=param.test_batch_size)
        dataloaders.append(("KITTI", video, dataloader))

    return dataloaders

def _load_midair_test_trajectories_dataloaders(param: Parameters) -> list:
    """
    Will load each MidAir test trajectories defined in the parameters into its own dataloader.
    The trajectory will be segmented into multiple segments of size param.sliding_window_size, with
    an overlap of param.sliding_window_overlap.
    @param param:
    @return: A list of tuples ("MidAir", trajectory name, dataloader). One for each trajectory
    """
    dataloaders = []

    for trajectory in param.midair_test_trajectories:
        midair_test_dataset = MidAirImageSequenceDatasetEulerDifferences(param.midair_test_path,
                                                                         new_size=(param.img_w, param.img_h),
                                                                         img_mean=param.midair_mean,
                                                                         img_std=param.midair_std,
                                                                         resize_mode=param.resize_mode,
                                                                         minus_point_5=param.minus_point_5,
                                                                         trajectories=[trajectory],
                                                                         augment_dataset=param.test_dataset_augmentation)
        dataloader = DataLoader(midair_test_dataset, num_workers=0,
                              pin_memory=param.pin_mem, batch_size=param.test_batch_size)
        dataloaders.append(("MidAir", trajectory, dataloader))

    return dataloaders

def _load_kitti(par: Parameters) -> tuple:

    #Load the dataset by string from the parameters
    try:
        dataset_class = getattr(KITTI, "KITTI" + par.dataset_suffix)
    except:
        NotImplementedError("Dataset class {} does not exist. Please check the dataset name and dataset_suffix")
    CometLogger.print("Using specific dataset: {}".format("KITTI" + par.dataset_suffix))

    train_dataset = dataset_class(par.kitti_training_segments, new_size=(par.img_w, par.img_h),
                                  img_mean=par.kitti_mean, img_std=par.kitti_std,
                                  resize_mode=par.resize_mode, minus_point_5=par.minus_point_5,
                                  augment_dataset=par.training_dataset_augmentation)

    train_sampler = SortedRandomBatchSegmentSampler(dataset=train_dataset, batch_size=par.batch_size, drop_last=par.drop_last_extra_segment)
    train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=par.n_processors,
                          pin_memory=par.pin_mem)

    valid_dataset = dataset_class(par.kitti_validation_segments, new_size=(par.img_w, par.img_h),
                                               img_mean=par.kitti_mean, img_std=par.kitti_std,
                                               resize_mode=par.resize_mode, minus_point_5=par.minus_point_5)

    valid_sampler = SortedRandomBatchSegmentSampler(dataset=valid_dataset, batch_size=par.batch_size, drop_last=par.drop_last_extra_segment)
    valid_dl = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=par.n_processors,
                          pin_memory=par.pin_mem)

    return train_dataset, train_dl, valid_dataset, valid_dl

def segment_datasets(param:Parameters):
    if param.dataset == "KITTI":
        segment_KITTI(param)
    elif param.dataset == "MidAir":
        segment_midair(param)
    elif param.dataset == "all":
        segment_KITTI(param)
        segment_midair(param)
    else:
        raise NotImplementedError("Dataset segmentation for {}, is not implemented".format(param.dataset))


def segment_KITTI(par:Parameters):
    print("Segmenting the KITTI dataset")
    train_segmenter = KITTIDataSegmenter(folder_list=par.kitti_train_video, pose_dir=par.kitti_pose_dir,
                                         image_dir=par.kitti_image_dir,
                                         segments_destination=par.kitti_training_segments)
    train_segmenter.segment(seq_len_range=par.seq_len, overlap=par.overlap,
                            sample_times=par.sample_times)
    valid_segmenter = KITTIDataSegmenter(folder_list=par.kitti_valid_video, pose_dir=par.kitti_pose_dir,
                                         image_dir=par.kitti_image_dir,
                                         segments_destination=par.kitti_validation_segments)
    valid_segmenter.segment(seq_len_range=par.seq_len, overlap=par.overlap,
                            sample_times=par.sample_times)
    for video in par.kitti_test_video:
        segment_destination = par.kitti_test_segments_directory+"/{}.pickle".format(video)
        test_segmenter = KITTIDataSegmenter(folder_list=[video], pose_dir=par.kitti_pose_dir,
                                                  image_dir=par.kitti_image_dir,
                                                  segments_destination=segment_destination)
        test_segmenter.segment(seq_len_range=(par.sliding_window_size,), overlap=par.sliding_window_overlap)

def segment_midair(param:Parameters):
    print("Segmenting the MidAir dataset")
    train_data_segmenter = MidAirDataSegmenter(param.midair_training_path)
    train_data_segmenter.segment(param.seq_len, overlap=param.overlap)
    valid_data_segmenter = MidAirDataSegmenter(param.midair_validation_path)
    valid_data_segmenter.segment(param.seq_len, overlap=param.overlap)
    test_data_segmenter = MidAirDataSegmenter(param.midair_test_path)
    test_data_segmenter.segment((param.sliding_window_size,), overlap=param.sliding_window_overlap)

def _load_midAir_dataset(param: Parameters) -> tuple:

    #Load the dataset by string from the parameters
    try:
        dataset_class = getattr(MidAir, "MidAir" + param.dataset_suffix)
    except:
        NotImplementedError("Dataset class {} does not exist. Please check the dataset name and dataset_suffix")

    CometLogger.print("Using specific dataset: {}".format("MidAir" + param.dataset_suffix))

    train_dataset = dataset_class(param.midair_training_path,
                                  new_size=(param.img_w, param.img_h),
                                  img_mean=param.midair_mean, img_std=param.midair_std,
                                  resize_mode=param.resize_mode,
                                  minus_point_5=param.minus_point_5,
                                  augment_dataset=param.training_dataset_augmentation)
    train_random_sampler = SortedRandomBatchSegmentSampler(dataset=train_dataset, batch_size=param.batch_size, drop_last=param.drop_last_extra_segment)
    valid_dataset = dataset_class(param.midair_validation_path,
                                                               new_size=(param.img_w, param.img_h),
                                                               img_mean=param.midair_mean, img_std=param.midair_std,
                                                               resize_mode=param.resize_mode,
                                                               minus_point_5=param.minus_point_5)
    valid_random_sampler = SortedRandomBatchSegmentSampler(dataset=valid_dataset, batch_size=param.batch_size, drop_last=param.drop_last_extra_segment)
    train_dataloader = DataLoader(train_dataset, num_workers=param.n_processors,
                          pin_memory=param.pin_mem, batch_sampler=train_random_sampler)
    valid_dataloader = DataLoader(valid_dataset, num_workers=param.n_processors,
                          pin_memory=param.pin_mem, batch_sampler=valid_random_sampler)

    return train_dataset, train_dataloader, valid_dataset, valid_dataloader

def _load_all_datasets(param: Parameters) -> tuple:
    kitti_train_dataset, _, kitti_valid_dataset, _ = _load_kitti(param)
    midair_train_dataset, _, midair_valid_dataset, _ = _load_midAir_dataset(param)

    train_dataset = kitti_train_dataset + midair_train_dataset
    valid_dataset = kitti_valid_dataset + midair_valid_dataset

    train_sampler = SortedRandomBatchSegmentSampler(dataset=train_dataset, batch_size=param.batch_size, drop_last=param.drop_last_extra_segment)
    train_dl = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=param.n_processors,
                          pin_memory=param.pin_mem)

    valid_sampler = SortedRandomBatchSegmentSampler(dataset=valid_dataset, batch_size=param.batch_size, drop_last=param.drop_last_extra_segment)
    valid_dl = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=param.n_processors,
                          pin_memory=param.pin_mem)

    return train_dataset, train_dl, valid_dataset, valid_dl


def load_model(param: Parameters) -> nn.Module:
    if param.model == "DeepVO":
        CometLogger.print("Using DeepVO")
        model = DeepVO(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "CoordConvDeepVO":
        CometLogger.print("Using CoordConvDeepVO")
        model = CoordConvDeepVO(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "MagicVO":
        CometLogger.print("Using MagicVO")
        model = MagicVO(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "SelfAttentionVO":
        CometLogger.print("Using SelfAttentionVO")
        model = SelfAttentionVO(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "SplitSelfAttentionVO":
        CometLogger.print("Using SplitSelfAttentionVO")
        model = SplitSelfAttentionVO(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "CoordConvSelfAttentionVO":
        CometLogger.print("Using CoordConvSelfAttentionVO")
        model = CoordConvSelfAttentionVO(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "SimpleSelfAttentionVO":
        CometLogger.print("Using SimpleSelfAttentionVO")
        model = SimpleSelfAttentionVO(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "PositionalSimpleSelfAttentionVO":
        CometLogger.print("Using PositionalSimpleSelfAttentionVO")
        model = PositionalSimpleSelfAttentionVO(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "SkippedSelfAttention":
        CometLogger.print("Using SkippedSelfAttention")
        model = SkippedSelfAttention(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "WeightedSelfAttentionVO":
        CometLogger.print("Using WeightedSelfAttentionVO")
        model = WeightedSelfAttentionVO(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "SelfAttentionVO_GlobRelOutput":
        CometLogger.print("Using SelfAttentionVO_GlobRelOutput")
        model = SelfAttentionVO_GlobRelOutput(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "StackedSelfAttentionVO":
        CometLogger.print("Using StackedSelfAttentionVO")
        model = StackedSelfAttentionVO(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "NoSelfAttentionVO":
        CometLogger.print("Using NoSelfAttentionVO")
        model = NoSelfAttentionVO(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "SnailSelfAttentionVO":
        CometLogger.print("Using SnailSelfAttentionVO")
        model = SnailSelfAttentionVO(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "SnailVO":
        CometLogger.print("Using SnailSelfAttentionVO")
        model = SnailVO(param.img_h, param.img_w, 5)
    elif param.model == "GlobalRelativeSelfAttentionVO":
        CometLogger.print("Using GlobalRelativeSelfAttentionVO")
        model = GlobalRelativeSelfAttentionVO(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "GlobalRelativeTransformerVO":
        CometLogger.print("Using GlobalRelativeTransformerVO")
        model = GlobalRelativeTransformerVO(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "GlobalRelativeTransformerVO_globXAsKeyVal":
        CometLogger.print("Using GlobalRelativeTransformerVO_globXAsKeyVal")
        model = GlobalRelativeTransformerVO_globXAsKeyVal(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    elif param.model == "GlobalRelativeSelfAttentionVO_globXasKeyVal":
        CometLogger.print("Using GlobalRelativeSelfAttentionVO_globXasKeyVal")
        model = GlobalRelativeSelfAttentionVO_globXasKeyVal(param.img_h, param.img_w, rnn_hidden_size=param.rnn_hidden_size)
    else:
        CometLogger.print("{} was not implemented".format(param.model))
        raise NotImplementedError()

    _map_pretrained_model_to_current_model(param.pretrained_model, model)

    if cuda_is_available():
        CometLogger.print("Training with CUDA")
        model.cuda()
    else:
        CometLogger.print("CUDA not available. Training on the CPU.")

    return model


def _map_pretrained_model_to_current_model(pretrained_model_path: str, model: nn.Module):
    CometLogger.print("Loading pretrain model: {}".format(pretrained_model_path))
    pretrained_model = torch.load(pretrained_model_path, map_location='cpu')
    try:
        model.load_state_dict(pretrained_model)
    except:
        model_dict = model.state_dict()
        # Will map values of common keys only
        common_updated_dict = {k: v for k, v in pretrained_model['state_dict'].items() if k in model_dict}
        model_dict.update(common_updated_dict)
        model.load_state_dict(model_dict)

def load_optimizer(param: Parameters, model: nn.Module) -> Optimizer:
    CometLogger.get_experiment().log_parameter("Optimizer", param.optimizer)
    CometLogger.get_experiment().log_parameter("Learning rate", param.learning_rate)

    if param.optimizer is "Adagrad":
        CometLogger.print("Using Adagrad")
        return optim.Adagrad(model.parameters(), lr=param.learning_rate)
    elif param.optimizer is "Adam":
        CometLogger.print("Using Adam Optimizer")
        return optim.Adam(model.parameters(), lr=param.learning_rate)
    elif param.optimizer is "RMSProp":
        CometLogger.print("Using RMSProp Optimizer")
        return optim.RMSprop(model.parameters(), lr=param.learning_rate)
    else:
        CometLogger.print("Optimizer {} was not implemented".format(param.optimizer))
        raise NotImplementedError()


def load_loss(param: Parameters) -> AbstractLoss:
    # Load the loss by string from the parameters
    loss_class = getattr(Losses, param.loss_function)
    initialization_arguments = loss_class.__init__.__code__.co_varnames
    arguments = dict()

    if len(initialization_arguments) > 0:
        for k in initialization_arguments:
            if k in param.get_params_as_dict():
                arguments[k] = param.get_params_as_dict()[k]

    CometLogger.print("Using loss: {}".format(param.loss_function))
    return loss_class(**arguments)