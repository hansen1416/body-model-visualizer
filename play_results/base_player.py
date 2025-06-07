import os
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np
import open3d as o3d
from smplx import SMPL, SMPLX, MANO, FLAME

from utils import (
    get_checkerboard_plane,
)


class BasePlayer(ABC):

    def __init__(self, result_path, video_path):

        # get abs path of current file
        abs_path = os.path.dirname(os.path.abspath(__file__))

        self.smpl_model = SMPL(
            os.path.join(
                abs_path, "..", "data", "body_models", "smpl", "SMPL_NEUTRAL.pkl"
            )
        )

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        self.total_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.vertices = self.load_result(result_path)

        # assert self.total_frame_count == len(
        #     self.vertices
        # ), f"Frame count mismatch, {self.total_frame_count} != {len(self.vertices)}"

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.cap.release()

        self.step = 1 / fps

    @abstractmethod
    def load_result(self, result_path) -> np.ndarray:
        pass

    def load_smpl_mesh(self, verts, faces):

        mesh = o3d.geometry.TriangleMesh()

        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.5, 0.5, 0.5])

        return mesh

    def play(self):

        print(f"Total frames: {self.total_frame_count}, FPS: {1 / self.step}")

        frame_idx = 0

        faces = self.smpl_model.faces

        mesh = self.load_smpl_mesh(self.vertices[frame_idx], faces)

        self.vis.add_geometry(mesh)

        # also need the graound
        gp = get_checkerboard_plane(plane_width=2, num_boxes=9)

        for _, g in enumerate(gp):
            g.compute_vertex_normals()
            self.vis.add_geometry(g)

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

            mesh.vertices = o3d.utility.Vector3dVector(self.vertices[frame_idx])
            # self.vis.add_geometry(mesh)
            self.vis.update_geometry(mesh)

            self.vis.poll_events()
            self.vis.update_renderer()

            frame_idx += 1

            time.sleep(self.step)

        self.vis.destroy_window()  # Close the visualizer
