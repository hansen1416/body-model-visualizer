import os
import joblib
import time

import cv2
import boto3
import numpy as np
import torch
import open3d as o3d
from smplx import SMPL, SMPLX, MANO, FLAME


class WHAMPlayer:

    def __init__(self):

        video_name = "5 Dumbbell HIIT exercises you need to add!"

        wham_result_path = os.path.join(
            os.path.expanduser("~"),
            "Downloads",
            "wham-results",
            video_name,
            "wham_output.pkl",
        )

        wham_result = joblib.load(wham_result_path)

        # pose = wham_result[0]["pose"]
        # beta = wham_result[0]["betas"]
        self.verts = wham_result[0]["verts"]

        self.verts[:, :, 1] *= -1
        self.verts[:, :, 2] *= -1

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.smpl_model = SMPL(
            os.path.join("data", "body_models", "smpl", "SMPL_NEUTRAL.pkl")
        )

        self.smpl_frame = 0
        self.total_frame = len(self.verts)

        video_path = os.path.join(
            os.path.expanduser("~"),
            "Downloads",
            "videos",
            f"{video_name}.mp4",
        )

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

    def load_smpl_mesh(self, verts, faces):

        mesh = o3d.geometry.TriangleMesh()

        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.5, 0.5, 0.5])

        return mesh

    def play(self):

        faces = self.smpl_model.faces

        mesh = self.load_smpl_mesh(self.verts[self.smpl_frame], faces)

        self.vis.add_geometry(mesh)

        while True:

            mesh.vertices = o3d.utility.Vector3dVector(self.verts[self.smpl_frame])

            self.vis.update_geometry(mesh)
            self.vis.poll_events()
            self.vis.update_renderer()

            self.smpl_frame += 1

            if self.smpl_frame >= self.total_frame:
                break

            time.sleep(0.016)

        self.vis.destroy_window()  # Close the visualizer


if __name__ == "__main__":

    player = WHAMPlayer()
    player.play()
    player.close()
