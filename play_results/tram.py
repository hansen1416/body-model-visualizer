import os
import joblib
import time

import cv2
import numpy as np
import open3d as o3d
from smplx import SMPL, SMPLX, MANO, FLAME


class TramPlayer:

    def __init__(self, tram_result_path, video_path):

        tram_result = np.load(tram_result_path, allow_pickle=True)

        # pose = wham_result[0]["pose"]
        # beta = wham_result[0]["betas"]
        print(tram_result)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        # get abs path of current file
        abs_path = os.path.dirname(os.path.abspath(__file__))

        self.smpl_model = SMPL(
            os.path.join(
                abs_path, "..", "data", "body_models", "smpl", "SMPL_NEUTRAL.pkl"
            )
        )

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        self.total_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        assert self.total_frame_count == len(self.verts), "Frame count mismatch"

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.cap.release()

        self.step = 1 / fps

    def load_smpl_mesh(self, verts, faces):

        mesh = o3d.geometry.TriangleMesh()

        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.5, 0.5, 0.5])

        return mesh

    def play(self):

        frame_idx = 0

        faces = self.smpl_model.faces

        mesh = self.load_smpl_mesh(self.verts[frame_idx], faces)

        self.vis.add_geometry(mesh)

        # image_geometry = None

        while frame_idx < self.total_frame_count:

            # # Convert the frame from BGR to RGB
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # # Create an Open3D image from the frame
            # image_geometry = o3d.geometry.Image(frame)

            # # only remove image_geometry from self.vis
            # # self.vis.remove_geometry(image_geometry)
            # self.vis.clear_geometries()
            # self.vis.add_geometry(image_geometry)
            # self.vis.update_geometry(image_geometry)

            mesh.vertices = o3d.utility.Vector3dVector(self.verts[frame_idx])
            # self.vis.add_geometry(mesh)
            self.vis.update_geometry(mesh)

            self.vis.poll_events()
            self.vis.update_renderer()

            frame_idx += 1

            time.sleep(self.step)

        self.vis.destroy_window()  # Close the visualizer


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

        break
