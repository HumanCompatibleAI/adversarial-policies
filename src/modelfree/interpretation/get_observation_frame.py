import logging
import os
import re

import cv2
from tqdm import tqdm

from modelfree.interpretation.visualize_tsne import _get_latest_sacred_dir_with_params

logger = logging.getLogger('modelfree.interpretation.get_observation_frame')
base_path = "/Users/cody/Data/adversarial_policies/tsne_save_activations/"
output_path = "/Users/cody/Data/adversarial_policies/video_frames/"


def get_frames(opponent_type, sacred_id=None):
    if sacred_id is None:
        sacred_id = _get_latest_sacred_dir_with_params(os.path.join(base_path, opponent_type))
    sacred_video_path = os.path.join(base_path, opponent_type, sacred_id, 'videos')
    video_paths = os.listdir(sacred_video_path)
    video_paths = [fp for fp in video_paths if 'mp4' in fp]
    ptn = re.compile(r'episode_(\d+)_recording.mp4')
    episodes = [ptn.search(fp).groups()[0] for fp in video_paths]
    sacred_path = os.path.join(output_path, opponent_type, str(sacred_id))
    if not os.path.exists(sacred_path):
        os.mkdir(sacred_path)

    for episode, path in tqdm(list(zip(episodes, video_paths))):
        episode_path = os.path.join(sacred_path, str(episode))
        if not os.path.exists(episode_path):
            os.mkdir(episode_path)
        vid = cv2.VideoCapture(os.path.join(sacred_video_path, path))
        frames_seen = 0
        while True:
            success, frame = vid.read()
            if not success:
                break
            # if frames_seen % 10 == 0:
            #     print("Frame {} processed".format(frames_seen))
            file_name = "{}_episode_{}_frame_{}.jpg".format(opponent_type, episode, frames_seen)
            cv2.imwrite(os.path.join(episode_path, file_name), frame)
            frames_seen += 1


if __name__ == "__main__":
    logger.info("Parsing adversary videos")
    get_frames(opponent_type='adversary')
    logger.info("Parsing random videos")
    get_frames(opponent_type='random')
    logger.info("Parsing zoo videos")
    get_frames(opponent_type='zoo')
