import os
import os.path as osp
import joblib

import cv2
import torch
import imageio
import numpy as np
from progress.bar import Bar
from smplx import SMPL, SMPLX, MANO, FLAME

from renderer import Renderer


# def run_vis_on_demo(video, results, output_pth, smpl, vis_global=True):
def run_vis_on_demo(video, results, output_pth, smpl):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # to torch tensor
    tt = lambda x: torch.from_numpy(x).float().to(device)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )

    # create renderer with cliff focal length estimation
    focal_length = (width**2 + height**2) ** 0.5
    renderer = Renderer(width, height, focal_length, device, smpl.faces)

    # if vis_global:
    #     # setup global coordinate subject
    #     # current implementation only visualize the subject appeared longest
    #     n_frames = {k: len(results[k]["frame_ids"]) for k in results.keys()}
    #     sid = max(n_frames, key=n_frames.get)
    #     global_output = smpl.get_output(
    #         body_pose=tt(results[sid]["pose_world"][:, 3:]),
    #         global_orient=tt(results[sid]["pose_world"][:, :3]),
    #         betas=tt(results[sid]["betas"]),
    #         transl=tt(results[sid]["trans_world"]),
    #     )
    #     verts_glob = global_output.vertices.cpu()
    #     verts_glob[..., 1] = verts_glob[..., 1] - verts_glob[..., 1].min()
    #     cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[
    #         [0, 2]
    #     ] / 2.0
    #     sx, sz = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]]
    #     scale = max(sx.item(), sz.item()) * 1.5

    #     # set default ground
    #     renderer.set_ground(scale, cx.item(), cz.item())

    #     # build global camera
    #     global_R, global_T, global_lights = get_global_cameras(verts_glob, device)

    # build default camera
    default_R, default_T = torch.eye(3), torch.zeros(3)

    writer = imageio.get_writer(
        osp.join(output_pth, "output.mp4"),
        fps=fps,
        mode="I",
        format="FFMPEG",
        macro_block_size=1,
    )
    bar = Bar("Rendering results ...", fill="#", max=length)

    frame_i = 0
    # _global_R, _global_T = None, None
    # run rendering
    while cap.isOpened():
        flag, org_img = cap.read()
        if not flag:
            break
        img = org_img[..., ::-1].copy()

        # render onto the input video
        renderer.create_camera(default_R, default_T)
        for _id, val in results.items():
            # render onto the image
            frame_i2 = np.where(val["frame_ids"] == frame_i)[0]
            if len(frame_i2) == 0:
                continue
            frame_i2 = frame_i2[0]
            img = renderer.render_mesh(
                torch.from_numpy(val["verts"][frame_i2]).to(device), img
            )

        # if vis_global:
        #     # render the global coordinate
        #     if frame_i in results[sid]["frame_ids"]:
        #         frame_i3 = np.where(results[sid]["frame_ids"] == frame_i)[0]
        #         verts = verts_glob[[frame_i3]].to(device)
        #         faces = renderer.faces.clone().squeeze(0)
        #         colors = torch.ones((1, 4)).float().to(device)
        #         colors[..., :3] *= 0.9

        #         if _global_R is None:
        #             _global_R = global_R[frame_i3].clone()
        #             _global_T = global_T[frame_i3].clone()
        #         cameras = renderer.create_camera(global_R[frame_i3], global_T[frame_i3])
        #         img_glob = renderer.render_with_ground(
        #             verts, faces, colors, cameras, global_lights
        #         )

        #     try:
        #         img = np.concatenate((img, img_glob), axis=1)
        #     except:
        #         img = np.concatenate((img, np.ones_like(img) * 255), axis=1)

        writer.append_data(img)
        bar.next()
        frame_i += 1
    writer.close()


if __name__ == "__main__":

    video_dir = os.path.join(os.path.expanduser("~"), "Downloads", "videos")

    # iterate over all videos in the directory
    for video_file in os.listdir(video_dir):

        print(f"Processing video: {video_file}")

        filename = video_file.split(".")[0]

        video_path = os.path.join(video_dir, video_file)
        wham_result_path = os.path.join(
            os.path.expanduser("~"),
            "Downloads",
            "wham-results",
            filename,
            "wham_output.pkl",
        )

        if not os.path.exists(wham_result_path):
            print(f"WHAM result not found for {video_file}")
            continue
            # WHAM result not found for Top 10 Intense Bodyweight Cardio.mp4
            # WHAM result not found for 🔥 300 Rep Bodyweight HIIT Workout.mp4

        output_path = os.path.join(
            os.path.expanduser("~"),
            "Downloads",
            "wham-results",
            filename,
        )

        smpl_model = SMPL(
            os.path.join("data", "body_models", "smpl", "SMPL_NEUTRAL.pkl")
        )

        wham_result = joblib.load(wham_result_path)

        # load the video and run the visualization

        run_vis_on_demo(video_path, wham_result, output_path, smpl_model)
