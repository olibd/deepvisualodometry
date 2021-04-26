import pickle

# Original inspiration for using a parameter class came from https://github.com/ChiWeiHsiao/DeepVO-pytorch
class Parameters():
    force_run_on_cpu = False

    def __init__(self):
        self.dataset = "KITTI"  # options MidAir, KITTI, all
        self.dataset_suffix = "ImageSequenceDatasetEulerDifferences"
        self.training_dataset_augmentation = False  # Randomly applied color jitters and cutouts
        self.test_dataset_augmentation = False  # Randomly applied color jitters and cutouts
        self.model = "SelfAttentionVO"  # options DeepVO, MagicVO, SelfAttentionVO, StackedSelfAttentionVO, SnailSelfAttentionVO, SnailVO, GlobalRelativeSelfAttentionVO,
        self.segment_dataset = False
        self.resume = False
        self.train = True
        self.test = True

        ######## Data Config ########
        self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
        self.img_w = 608  # original size is about 1226
        self.img_h = 184  # original size is about 370
        self.minus_point_5 = True
        self.seq_len = (5, 7)
        self.sample_times = 3
        self.overlap = 1
        self.n_processors = 4  # nbr of cpu processing cores

        ######## Model Config ########
        #self.pretrained_model = "/home/olivier/Documents/candidate_models/SelfAttentionVO_Augmented_KITTI_and_MidAir/SelfAttentionVO__MultiDataset_Adagrad_BatchSegmentMSELoss_model.checkpoint"
        self.pretrained_model = "./pretrained/flownets_EPE1.951.pth.tar"
        self.rnn_hidden_size = 1000

        ######## Training Config ########
        self.epochs = 250
        self.batch_size = 7
        self.drop_last_extra_segment = True  # If the last segment doesn't fit in the batch size, do not use it in a batch.
        self.pin_mem = False
        self.optimizer = 'Adagrad'
        self.learning_rate = 0.0005
        self.loss_function = "BatchSegmentMSELoss"  # Class name of the loss, losses are located in Model.Losses
        self.early_stopping_patience = 15
        self.model_backup_destination = "./model_backups"
        self.gradient_clipping_value = None

        ######## Testing Config ########
        """ 
        The number of frames seen in one inference pass by the algorithm.
        This number is analogous to the seq_len used in training.
        The test sequence will be batched into segments of the size of the sliding window
        In an RNN based model, this help preserves the data of the hidden space over long sequence since
        it is loss for each new inference pass. In temporal convolutional, based model it defines the size
        of the context seen in a forward pass by the network."""
        self.sliding_window_size = 30
        """The size of the sliding window overlap defines how much each frames will be shared between each
        inference pass. This allows for the context to be shared between each pass. The less overlap there is,
        the more new frames will be processed in one batch, so you'll go through a trajectory faster, but your
        context will be smaller."""
        self.sliding_window_overlap = 15
        assert self.sliding_window_size > self.sliding_window_overlap

        self.test_batch_size = 1

        ######## KITTI ########
        self.kitti_path = '/home/olivier/Documents/KITTI'
        self.kitti_image_dir = self.kitti_path + '/images/'
        self.kitti_pose_dir = self.kitti_path + '/pose_GT/dataset/poses/'
        self.kitti_train_video = ['00', '01', '02', '05', '08', '09']
        self.kitti_valid_video = ['04', '06']
        self.kitti_test_video = ['07', '10']
        self.kitti_training_segments = self.kitti_path + "/train_segment.pickle"
        self.kitti_validation_segments = self.kitti_path + "/validation_segment.pickle"
        self.kitti_test_segments_directory = self.kitti_path + "/test_segments"
        try:
            self.kitti_mean = [0.35034978101920455,0.36983621966485647,0.36422201692803685] #pickle.load(open(self.kitti_path + "/Means.pkl", "rb"))["mean_tensor"]
        except:
            print("WARNING: KITTI mean file not found, using 0. Run preprocess_datasets.py to generate the file")
            self.kitti_mean = [0, 0, 0]
        try:
            self.kitti_std = [0.3153969452509568,0.31948839594698725,0.3234648542300365]#pickle.load(open(self.kitti_path + "/StandardDevs.pkl", "rb"))["std_tensor"]
        except:
            print("WARNING: KITTI std dev file not found, using 0. Run preprocess_datasets.py to generate the file")
            self.kitti_std = [0, 0, 0]

        ######## MidAir ########
        self.midair_path = "/media/olivier/OS/Ubuntu/MidAir"
        self.midair_training_path = self.midair_path + "/MidAir_train"
        self.midair_validation_path = self.midair_path + "/MidAir_valid"
        self.midair_test_path = self.midair_path + "/VO_test"
        self.midair_test_trajectories = ["trajectory_0000", "trajectory_0001", "trajectory_0002",
                                         "trajectory_1000", "trajectory_1001", "trajectory_1002",
                                         "trajectory_2000", "trajectory_2001", "trajectory_2002"]
        try:
            self.midair_mean = pickle.load(open(self.midair_path + "/Means.pkl", "rb"))["mean_tensor"]
        except:
            print("WARNING: MidAir mean file not found, using 0. Run preprocess_datasets.py to generate the file")
            self.midair_mean = [0, 0, 0]
        try:
            self.midair_std = pickle.load(open(self.midair_path + "/StandardDevs.pkl", "rb"))["std_tensor"]
        except:
            print("WARNING: MidAir std dev file not found, using 0. Run preprocess_datasets.py to generate the file")
            self.midair_std = [0, 0, 0]

    def get_params_as_dict(self) -> dict:
        return self.__dict__