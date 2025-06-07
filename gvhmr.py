import os

import joblib
import numpy as np
import torch
import smplx

from utils.base_player import BasePlayer

# from body_model_smplx import BodyModelSMPLX


# def make_smplx(**kwargs):

#     # SuperMotion is trained on BEDLAM dataset, the smplx config is the same except only 10 betas are used
#     bm_kwargs = {
#         "model_type": "smplx",
#         "gender": "neutral",
#         "num_pca_comps": 12,
#         "flat_hand_mean": False,
#     }
#     bm_kwargs.update(kwargs)
#     model = BodyModelSMPLX(
#         model_path="/home/hlz/repos/t2hm-dataset/inputs/checkpoints/body_models",
#         **bm_kwargs,
#     )

#     return model


def get_ground_params_from_points(root_points, vert_points):
    """xz-plane is the ground plane
    Args:
        root_points: (L, 3), to decide center
        vert_points: (L, V, 3), to decide scale
    """
    # root_max = root_points.max(0)[0]  # (3,)
    # root_min = root_points.min(0)[0]  # (3,)
    root_max = np.max(root_points, axis=0)  # shape: (24, 3)
    root_min = np.min(root_points, axis=0)  # shape: (24, 3)

    cx, _, cz = (root_max + root_min) / 2.0

    # vert_max = vert_points.reshape(-1, 3).max(0)[0]  # (L, 3)
    # vert_min = vert_points.reshape(-1, 3).min(0)[0]  # (L, 3)
    vert_max = vert_points.reshape(-1, 3).max(axis=0)  # (3,)
    vert_min = vert_points.reshape(-1, 3).min(axis=0)  # (3,)

    scale = (vert_max - vert_min)[[0, 2]].max()
    return float(scale), float(cx), float(cz)


class GVHMRPlayer(BasePlayer):

    def __init__(self, result_path, video_path):

        super(GVHMRPlayer, self).__init__(result_path, video_path)

    def load_result(self, results) -> np.ndarray:

        # 'body_pose', 'betas', 'global_orient', 'transl'

        # print(results["body_pose"].shape)
        # print(results["betas"].shape)
        # print(results["global_orient"].shape)
        # print(results["transl"].shape)

        # pred = make_smplx()(**results)

        # pred = self.smpl_model(
        #     body_pose=results["body_pose"],
        #     global_orient=results["global_orient"],
        #     betas=results["betas"],
        #     transl=results["transl"],
        #     pose2rot=False,
        #     default_smpl=True,
        # )

        return results


if __name__ == "__main__":

    results_folder = os.path.join(
        os.path.expanduser("~"), "repos", "t2hm-dataset", "outputs", "demo"
    )
    # iterate over results folder

    for video_name in os.listdir(results_folder):

        joints_glob = torch.load(
            os.path.join(
                results_folder,
                video_name,
                "joints_glob.pt",
            )
        )

        verts_glob = torch.load(
            os.path.join(
                results_folder,
                video_name,
                "verts_glob.pt",
            )
        )

        joints_glob = joints_glob.cpu().numpy()
        verts_glob = verts_glob.cpu().numpy()

        print(joints_glob.shape)
        print(verts_glob.shape)

        scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)

        print(f"Scale: {scale}, Center: ({cx}, {cz})")

        # print(joints_glob.shape)
        # print(verts_glob.shape)

        # data: dict = torch.load(hmr_result)
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
        player = GVHMRPlayer(verts_glob, video_path)
        player.play()

        break
