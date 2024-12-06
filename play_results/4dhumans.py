import os

import joblib
import numpy as np

from base_player import BasePlayer


class HumansPlayer(BasePlayer):

    def __init__(self, result_path, video_path):

        super(HumansPlayer, self).__init__(result_path, video_path)

    def load_result(self, result_path) -> np.ndarray:
        human_result = joblib.load(result_path)

        print(human_result)


if __name__ == "__main__":

    results_folder = os.path.join(
        os.path.expanduser("~"), "Downloads", "4dhumans-results"
    )
    # iterate over results folder

    for video_name in os.listdir(results_folder):

        wham_result_path = os.path.join(
            results_folder,
            video_name,
            f"demo_{video_name}.pkl",
        )

        video_path = os.path.join(
            os.path.expanduser("~"),
            "Downloads",
            "videos",
            f"{video_name}.mp4",
        )

        player = HumansPlayer(wham_result_path, video_path)
        player.play()

        break
