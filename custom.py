import os
import joblib
import time

import boto3
import numpy as np
import torch
import open3d as o3d
from smplx import SMPL, SMPLX, MANO, FLAME


class WHAMPlayer:

    def __init__(self):

        video_name = "5 Dumbbell HIIT exercises you need to add!"

        self.output_path = os.path.join(
            os.path.expanduser("~"),
            "Downloads",
            "wham-results",
            video_name,
            "wham_output.pkl",
        )
        self.video_path = os.path.join(
            os.path.expanduser("~"),
            "Downloads",
            "videos",
            f"{video_name}.mp4",
        )

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.smpl_model = SMPL(
            os.path.join("data", "body_models", "smpl", "SMPL_NEUTRAL.pkl")
        )

        self.smpl_frame = 0

    def load_smpl_mesh(self, verts, faces):

        mesh = o3d.geometry.TriangleMesh()

        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.5, 0.5, 0.5])

        return mesh

    def _load_wham_results(self):

        wham_result = joblib.load(self.output_path)

        # pose = wham_result[0]["pose"]
        # beta = wham_result[0]["betas"]
        verts = wham_result[0]["verts"]

        verts[:, :, 1] *= -1
        verts[:, :, 2] *= -1

        return verts

    def play_smpl_mesh(self):

        verts = self._load_wham_results()
        faces = self.smpl_model.faces

        mesh = self.load_smpl_mesh(verts[self.smpl_frame], faces)

        self.vis.add_geometry(mesh)

        while True:

            mesh.vertices = o3d.utility.Vector3dVector(verts[self.smpl_frame])

            self.vis.update_geometry(mesh)
            self.vis.poll_events()
            self.vis.update_renderer()

            self.smpl_frame += 1

            if self.smpl_frame >= len(verts):
                break

            time.sleep(0.016)

    def close(self):

        self.vis.destroy_window()  # Close the visualizer


if __name__ == "__main__":

    player = WHAMPlayer()
    player.play_smpl_mesh()
    player.close()

# # wham_result = joblib.load(os.path.join("data", "WHAM", "slam_results.pth"))
# wham_result = joblib.load(os.path.join("data", "WHAM", "wham_output.pkl"))

# pose = wham_result[0]["pose"]
# beta = wham_result[0]["betas"]
# verts = wham_result[0]["verts"]

# # print(pose.shape)
# # print(beta.shape)
# # print(verts.shape)

# # verts[:, :, 0] *= -1
# verts[:, :, 1] *= -1
# verts[:, :, 2] *= -1


# # beta_tensor = torch.tensor(beta).unsqueeze(0)

# # smpl_output = smpl_model(beta=beta_tensor)

# # vertices = smpl_output.vertices.detach().cpu().numpy().squeeze()

# # print(vertices.shape)

# # exit()

# # print(smpl_output)

# frame_idx = 0

# # verts = smpl_output.vertices[0].detach().numpy()
# vertices = verts[frame_idx]
# faces = smpl_model.faces

# mesh = o3d.geometry.TriangleMesh()

# mesh.vertices = o3d.utility.Vector3dVector(vertices)
# mesh.triangles = o3d.utility.Vector3iVector(faces)
# mesh.compute_vertex_normals()
# mesh.paint_uniform_color([0.5, 0.5, 0.5])

# #############

# # Create a visualizer
# vis = o3d.visualization.Visualizer()
# vis.create_window()

# # Add the mesh to the visualizer
# vis.add_geometry(mesh)

# # Animation loop
# for i in range(len(verts)):  # Number of frames
#     # Update the vertices (for example, create a wave effect)
#     vertices = verts[frame_idx]

#     mesh.vertices = o3d.utility.Vector3dVector(vertices)

#     # Update the mesh in the visualizer
#     vis.update_geometry(mesh)

#     # Optionally, you can update the visualizer's view
#     vis.poll_events()
#     vis.update_renderer()

#     frame_idx += 1

#     # Control the speed of the animation
#     time.sleep(0.016)

# # Close the visualizer
# vis.destroy_window()
