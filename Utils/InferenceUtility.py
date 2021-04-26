import sys
import time
from threading import Thread

import ffmpeg
import numpy
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, transforms

from Common.Helpers import cuda_is_available
from Common.Trajectory import Trajectory, TrajectoryPlotter
from Loaders import load_model
from Parameters import Parameters

"""
Modified from: https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md#tensorflow-streaming
"""

def get_image_transformer(par: Parameters) -> Compose:
    transform_ops = []
    if par.resize_mode == 'crop':
        transform_ops.append(transforms.CenterCrop((par.img_h, par.img_w)))
    elif par.resize_mode == 'rescale':
        transform_ops.append(transforms.Resize((par.img_h, par.img_w)))
    return transforms.Compose(transform_ops)


def inference_thread(param: Parameters, trajectory: Trajectory):
    global inference_started, fps
    image_transformer = get_image_transformer(param)
    model = load_model(param)

    print("When stream is ready, press enter:")
    for line in sys.stdin:
        if '\n' == line:
            break

    device = "cpu"
    if cuda_is_available():
        device = "cuda"

    video_input_reader = (
        ffmpeg
            .input(source)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(cmd=["ffmpeg", "-hide_banner", "-loglevel", "error"], pipe_stdout=True)
    )
    i = 0
    new_frames = param.sliding_window_size - param.sliding_window_overlap
    assert new_frames >= 1
    previous_frames = None
    while True:
        time_start = time.time()
        if previous_frames is None or len(previous_frames) < param.sliding_window_size:
            video_bytes_buffer = video_input_reader.stdout.read(width * height * 3)
        else:
            video_bytes_buffer = video_input_reader.stdout.read(new_frames * width * height * 3)

        if not video_bytes_buffer:
            break

        if previous_frames is None or len(previous_frames) < param.sliding_window_size:
            frames = numpy.frombuffer(video_bytes_buffer, numpy.uint8).reshape((1, height, width, 3))
        else:
            frames = numpy.frombuffer(video_bytes_buffer, numpy.uint8).reshape((new_frames, height, width, 3))

        if previous_frames is None:
            previous_frames = frames
        else:
            previous_frames = numpy.concatenate((previous_frames, frames))

        if len(previous_frames) < param.sliding_window_size:
            # Do not start inference yet, build frame count up to window size
            continue
        elif len(previous_frames) > param.sliding_window_size:
            # Keep frame count same size as window size
            previous_frames = previous_frames[new_frames:]
            assert len(previous_frames) == param.sliding_window_size

        frame_tensors = torch.Tensor(previous_frames).to(device).permute(0, 3, 1, 2).div(255)
        frame_tensors = image_transformer(frame_tensors)
        if param.minus_point_5:
            frame_tensors = frame_tensors - 0.5
        frame_tensors = torch.unsqueeze(frame_tensors, 0)
        prediction = torch.squeeze(model.forward(frame_tensors)).detach().cpu().numpy()
        trajectory.append(prediction)

        time_stop = time.time()
        total_time = time_stop - time_start
        fps = len(previous_frames) / total_time
        print(f"fps: {len(previous_frames) / total_time}")
        i = i + 1
        inference_started = True


if __name__ == '__main__':
    param = Parameters()
    source: str = "https://devstreaming-cdn.apple.com/videos/streaming/examples/bipbop_16x9/bipbop_16x9_variant.m3u8"
    width = 1920
    height = 1080
    param.sliding_window_size = 30
    param.sliding_window_overlap = 15
    trajectory = Trajectory(is_input_data_relative=True,
                            is_groundtruth=False,
                            sliding_window_size=param.sliding_window_size,
                            sliding_window_overlap=param.sliding_window_overlap)
    plotter = TrajectoryPlotter(trajectory_name="Live Trajectory Estimation",
                                dataset_name="Unknown Source",
                                model_name=param.model)
    inference_started = False
    fps = 0

    thread = Thread(target=inference_thread, args=(param, trajectory))
    thread.start()

    while thread.is_alive():
        if not inference_started:
            continue
        plotter.update_all_plots(predictions=trajectory.assembled_pose)
        plt.pause(0.05)

    thread.join()
    plt.show()