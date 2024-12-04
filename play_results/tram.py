import os

import numpy as np
from smplx import SMPL

from base_player import BasePlayer


class TramPlayer(BasePlayer):

    def __init__(self, tram_result_path, video_path):

        super(TramPlayer, self).__init__(tram_result_path, video_path)

    def load_result(self, result_path) -> np.ndarray:
        # dict_keys(['pred_cam', 'pred_pose', 'pred_shape', 'pred_rotmat', 'pred_trans', 'frame'])
        tram_result = np.load(result_path, allow_pickle=True).item()

        pred_rotmat = tram_result["pred_rotmat"]
        pred_shape = tram_result["pred_shape"]
        pred_trans = tram_result["pred_trans"]

        # smpl = SMPL()

        pred = self.smpl_model(
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, [0]],
            betas=pred_shape,
            transl=pred_trans.squeeze(),
            pose2rot=False,
            default_smpl=True,
        )

        vertices = pred.vertices

        vertices[:, :, 1] *= -1
        vertices[:, :, 2] *= -1

        return vertices


if __name__ == "__main__":

    results_folder = os.path.join(os.path.expanduser("~"), "Downloads", "tram-results")
    # iterate over results folder

    for video_name in os.listdir(results_folder):

        tram_result_path = os.path.join(
            results_folder, video_name, "hps", "hps_track_0.npy"
        )

        video_path = os.path.join(
            os.path.expanduser("~"),
            "Downloads",
            "videos",
            f"{video_name}.mp4",
        )

        player = TramPlayer(tram_result_path, video_path)
        player.play()

        break
