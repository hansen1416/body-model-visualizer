import os

import joblib
import numpy as np
import torch
import smplx

from base_player import BasePlayer

from body_model_smplx import BodyModelSMPLX


def make_smplx(**kwargs):

    # SuperMotion is trained on BEDLAM dataset, the smplx config is the same except only 10 betas are used
    bm_kwargs = {
        "model_type": "smplx",
        "gender": "neutral",
        "num_pca_comps": 12,
        "flat_hand_mean": False,
    }
    bm_kwargs.update(kwargs)
    model = BodyModelSMPLX(
        model_path="/home/hlz/repos/t2hm-dataset/inputs/checkpoints/body_models",
        **bm_kwargs,
    )

    return model


class GVHMRPlayer(BasePlayer):

    def __init__(self, result_path, video_path):

        super(GVHMRPlayer, self).__init__(result_path, video_path)

    def load_result(self, results) -> np.ndarray:

        # 'body_pose', 'betas', 'global_orient', 'transl'

        print(results["body_pose"].shape)
        print(results["betas"].shape)
        print(results["global_orient"].shape)
        print(results["transl"].shape)

        pred = make_smplx()(**results)

        # pred = self.smpl_model(
        #     body_pose=results["body_pose"],
        #     global_orient=results["global_orient"],
        #     betas=results["betas"],
        #     transl=results["transl"],
        #     pose2rot=False,
        #     default_smpl=True,
        # )

        vertices = pred.vertices

        # vertices[:, :, 1] *= -1
        # vertices[:, :, 2] *= -1

        return vertices


if __name__ == "__main__":

    results_folder = os.path.join(
        os.path.expanduser("~"), "repos", "t2hm-dataset", "outputs", "demo"
    )
    # iterate over results folder

    for video_name in os.listdir(results_folder):

        hmr_result = os.path.join(
            results_folder,
            video_name,
            "hmr4d_results.pt",
        )

        data: dict = torch.load(hmr_result)
        # print(data.keys())
        # dict_keys(['smpl_params_global', 'smpl_params_incam', 'K_fullimg', 'net_outputs'])

        # print(data["smpl_params_global"].keys())
        # dict_keys(['body_pose', 'betas', 'global_orient', 'transl'])

        # for k, v in data["smpl_params_global"].items():
        # print(f"{k}: {v.shape}")
        # body_pose: torch.Size([336, 63])
        # betas: torch.Size([336, 10])
        # global_orient: torch.Size([336, 3])
        # transl: torch.Size([336, 3])

        # print(data["smpl_params_incam"].keys())
        # dict_keys(['body_pose', 'betas', 'global_orient', 'transl'])

        # for k, v in data["smpl_params_incam"].items():
        #     print(f"{k}: {v.shape}")
        # body_pose: torch.Size([336, 63])
        # betas: torch.Size([336, 10])
        # global_orient: torch.Size([336, 3])
        # transl: torch.Size([336, 3])

        # print(data["K_fullimg"].shape)
        # torch.Size([336, 3, 3])

        # print(data["net_outputs"].keys())
        # dict_keys(['model_output', 'decode_dict', 'pred_smpl_params_incam', 'pred_smpl_params_global', 'static_conf_logits'])
        # this the full output of the network, including both 'smpl_params_global' and 'smpl_params_incam'
        # for more information, refer to hmr4d/model/gvhmr/gvhmr_pl_demo.py

        video_path = os.path.join(
            os.path.expanduser("~"),
            "Downloads",
            "videos",
            f"{video_name}.mp4",
        )

        # check if video file exists
        if not os.path.exists(video_path):
            print(f"Video file does not exist: {video_path}")
            continue

        # player = GVHMRPlayer(data["smpl_params_global"], video_path)
        player = GVHMRPlayer(data["smpl_params_incam"], video_path)
        player.play()

        break
