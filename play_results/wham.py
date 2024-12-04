import os

import joblib
import numpy as np

from base_player import BasePlayer


class WHAMPlayer(BasePlayer):

    def __init__(self, wham_result_path, video_path):

        super(WHAMPlayer, self).__init__(wham_result_path, video_path)

    def load_result(self, result_path) -> np.ndarray:
        wham_result = joblib.load(result_path)

        vertices = wham_result[0]["verts"]

        vertices[:, :, 1] *= -1
        vertices[:, :, 2] *= -1

        return vertices


if __name__ == "__main__":

    results_folder = os.path.join(os.path.expanduser("~"), "Downloads", "wham-results")
    # iterate over results folder

    for video_name in os.listdir(results_folder):

        wham_result_path = os.path.join(
            results_folder,
            video_name,
            "wham_output.pkl",
        )

        video_path = os.path.join(
            os.path.expanduser("~"),
            "Downloads",
            "videos",
            f"{video_name}.mp4",
        )

        player = WHAMPlayer(wham_result_path, video_path)
        player.play()

        break
