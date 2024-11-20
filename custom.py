import os
import joblib
import time

import numpy as np
import torch
import open3d as o3d
from smplx import SMPL, SMPLX, MANO, FLAME

# wham_result = joblib.load(os.path.join("data", "WHAM", "slam_results.pth"))
wham_result = joblib.load(os.path.join("data", "WHAM", "wham_output.pkl"))

pose = wham_result[0]["pose"]
beta = wham_result[0]["betas"]
verts = wham_result[0]["verts"]

print(pose.shape)
print(beta.shape)
print(verts.shape)


SMPL_PATH = os.path.join("data", "body_models", "smpl", "SMPL_NEUTRAL.pkl")

smpl_model = SMPL(SMPL_PATH)

# beta_tensor = torch.tensor(beta).unsqueeze(0)

# smpl_output = smpl_model(beta=beta_tensor)

# vertices = smpl_output.vertices.detach().cpu().numpy().squeeze()

# print(vertices.shape)

# exit()

# print(smpl_output)

frame_idx = 0

# verts = smpl_output.vertices[0].detach().numpy()
vertices = verts[frame_idx]
faces = smpl_model.faces

mesh = o3d.geometry.TriangleMesh()

mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.5, 0.5, 0.5])

#############

# Create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the mesh to the visualizer
vis.add_geometry(mesh)

# Animation loop
for i in range(len(verts)):  # Number of frames
    # Update the vertices (for example, create a wave effect)
    vertices = verts[frame_idx]

    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Update the mesh in the visualizer
    vis.update_geometry(mesh)

    # Optionally, you can update the visualizer's view
    vis.poll_events()
    vis.update_renderer()

    frame_idx += 1

    # Control the speed of the animation
    time.sleep(0.1)

# Close the visualizer
vis.destroy_window()
