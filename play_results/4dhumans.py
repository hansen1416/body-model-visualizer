import os
from typing import Optional
import warnings


import joblib
import numpy as np
import torch
from smplx import SMPLLayer
from smplx.lbs import vertices2joints
from smplx.utils import SMPLOutput


from base_player import BasePlayer

# Suppress all warnings
warnings.filterwarnings("ignore")


class SMPL(SMPLLayer):

    # @blockPrinting
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, **kwargs):
        """
        Extension of the official SMPL implementation to support more joints.
        Args:
            Same as SMPLLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """

        super(SMPL, self).__init__(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "data",
                "body_models",
                "smpl",
                "SMPL_NEUTRAL.pkl",
            ),
            *args,
            **kwargs,
        )
        smpl_to_openpose = [
            24,
            12,
            17,
            19,
            21,
            16,
            18,
            20,
            0,
            2,
            5,
            8,
            1,
            4,
            7,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
        ]

        # if joint_regressor_extra is not None:
        #     self.register_buffer(
        #         "joint_regressor_extra",
        #         torch.tensor(
        #             pickle.load(open(joint_regressor_extra, "rb"), encoding="latin1"),
        #             dtype=torch.float32,
        #         ),
        #     )
        self.register_buffer(
            "joint_map", torch.tensor(smpl_to_openpose, dtype=torch.long)
        )

    def forward(self, *args, **kwargs) -> SMPLOutput:
        """
        Run forward pass. Same as SMPL and also append an extra set of joints if joint_regressor_extra is specified.
        """
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        joints = smpl_output.joints[:, self.joint_map, :]
        # if hasattr(self, "joint_regressor_extra"):
        #     extra_joints = vertices2joints(
        #         self.joint_regressor_extra, smpl_output.vertices
        #     )
        #     joints = torch.cat([joints, extra_joints], dim=1)
        smpl_output.joints = joints
        return smpl_output


class HumansPlayer(BasePlayer):

    def __init__(self, result_path, video_path):

        super(HumansPlayer, self).__init__(result_path, video_path)

    def load_result(self, result_path) -> np.ndarray:

        vertices_path = result_path.replace(".pkl", ".npy")

        # if os.path.exists(vertices_path):
        #     return np.load(vertices_path)

        with open(result_path, "rb") as f:
            human_result = joblib.load(f)

        vertices = []

        for frame_idx, dict in enumerate(human_result.values()):

            if frame_idx >= 100:
                break

            # dict_keys(['global_orient', 'body_pose', 'betas'])
            smpl_info = dict["smpl"][0]

            # body_pose = torch.tensor(smpl_info["body_pose"], dtype=torch.float32)
            # global_orient = torch.tensor(
            #     smpl_info["global_orient"], dtype=torch.float32
            # )
            # betas = torch.tensor(smpl_info["betas"], dtype=torch.float32)

            #     body_pose.append(smpl_info["body_pose"])
            #     global_orient.append(smpl_info["global_orient"][:, 0])
            #     betas.append(smpl_info["betas"])

            body_pose = torch.tensor(smpl_info["body_pose"], dtype=torch.float32)
            global_orient = torch.tensor(
                smpl_info["global_orient"][:, 0], dtype=torch.float32
            )
            betas = torch.tensor(smpl_info["betas"], dtype=torch.float32)

            # body_pose = torch.tensor(body_pose, dtype=torch.float32)
            # global_orient = torch.tensor(global_orient, dtype=torch.float32)
            # betas = torch.tensor(betas, dtype=torch.float32)

            # print(body_pose.shape)
            # print(global_orient.shape)
            # print(betas.shape)

            # get vertices from SMPL
            smpl = SMPL(
                body_pose=body_pose,
                global_orient=global_orient,
                betas=betas,
                pose2rot=False,
            )

            smpl_output = smpl()

            vertices.append(smpl_output.vertices[0])

            # print(len(vertices))

        vertices = np.array(vertices)

        vertices[:, :, 1] *= -1
        vertices[:, :, 2] *= -1

        # save vertices to .npy at the same dir of `result_path`
        np.save(result_path.replace(".pkl", ".npy"), vertices)

        return vertices


if __name__ == "__main__":

    results_folder = os.path.join(
        os.path.expanduser("~"), "Downloads", "4dhumans-results"
    )
    # iterate over results folder

    for i, video_name in enumerate(os.listdir(results_folder)):

        wham_result_path = os.path.join(
            results_folder,
            video_name,
            "results",
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
