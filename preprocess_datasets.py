from Datasets.KITTI import KITTIDataPreprocessor
from Datasets.MidAir import MidAirDataPreprocessor
from Parameters import Parameters

if __name__ == '__main__':
    param = Parameters()
    if param.dataset is "MidAir" or param.dataset is "all":
        processor = MidAirDataPreprocessor(param.midair_path)
        print("Cleaning Midair dataset")
        processor.clean()
        print("Computing mean and std dev for the images of the MidAir dataset, this will take a while...")
        processor.compute_dataset_image_mean_std()
    if param.dataset is "KITTI" or param.dataset is "all":
        processor = KITTIDataPreprocessor(param.kitti_image_dir, param.kitti_pose_dir, param.kitti_path)
        print("Cleaning KITTI dataset")
        processor.clean()
        print("Computing mean and std dev for the images of the KITTI dataset, this will take a while...")
        processor.compute_dataset_image_mean_std()