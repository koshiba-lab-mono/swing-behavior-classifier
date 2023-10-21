import os
from pathlib import Path

VIDEO_TYPES = ["mp4", "avi", "MTS", "mov", "webm", "flv"]


def find_video_paths_in_directory(dir_path):
    video_paths: list[str] = []
    for VIDEO_TYPE in VIDEO_TYPES:
        video_paths.extend(list(dir_path.glob(f"*{VIDEO_TYPE}")))
    return video_paths


directory = Path("/app/videos")
video_paths = find_video_paths_in_directory(directory)

for video_path in video_paths:
    video_name = video_path.stem
    output_path = Path("/app/outputs") / video_path.stem
    os.makedirs(output_path, exist_ok=True)

    os.system(f"python object_tracker.py --video {video_path} --info {output_path} --dont_show --count")
