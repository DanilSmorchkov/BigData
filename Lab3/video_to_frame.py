import cv2
import os
from tqdm import tqdm


def all_frames(video_path):
    if not os.path.isdir('all_frames'):
        os.makedirs('all_frames')

    for directory in os.listdir(video_path):
        path = os.path.join(video_path, directory)
        video_to_frame(path, 'all_frames', 0.5, directory, name_video=True)


def train_frames_generation(input_path):
    if not os.path.exists('train/images'):
        os.makedirs('train/images')

    dirs = os.listdir(input_path)

    for directory in dirs:
        path = os.path.join(input_path, directory)
        video_to_frame(path, 'train/images', 0.5, directory, video_slices=slice(0, 5))


def validate_frames_generation(input_path):
    if not os.path.exists('val/images'):
        os.makedirs('val/images')

    dirs = os.listdir(input_path)

    for directory in dirs:
        path = os.path.join(input_path, directory)
        video_to_frame(path, 'val/images', 0.5, directory, video_slices=slice(5, 6))


def video_to_frame(input_dir: str, output_dir: str, time_step: float, name_dir: str, video_slices: slice | None = None,
                   name_video: bool = False) -> None:

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_frame_per_step = 0
    if video_slices is None:
        list_video = os.listdir(input_dir)
    else:
        list_video = os.listdir(input_dir)[video_slices]
    for video in tqdm(list_video, leave=False, position=0):
        video_path = os.path.join(input_dir, video)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = 0
        frames_in_time_step = int(time_step * fps)

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            if frame_count % frames_in_time_step == 0:
                if not name_video:
                    path = os.path.join(output_dir, f'{name_dir}_frame_{num_frame_per_step}.jpg')
                else:
                    path = os.path.join(output_dir, f'{name_dir}_{video[:-3]}_frame_{num_frame_per_step}.jpg')
                cv2.imwrite(path, frame)
                num_frame_per_step += 1
            frame_count += 1
        if name_video:
            num_frame_per_step = 0


if __name__ == '__main__':
    train_frames_generation('ITMO-HSE-MLBD-LW-3')
    validate_frames_generation('ITMO-HSE-MLBD-LW-3')
    all_frames('ITMO-HSE-MLBD-LW-3')
