import os
import joblib

import torch
import open3d as o3d
from smplx import SMPL, SMPLX, MANO, FLAME

# wham_result = joblib.load(os.path.join("data", "WHAM", "slam_results.pth"))
wham_result = joblib.load(os.path.join("data", "WHAM", "tracking_results.pth"))

print(wham_result)

exit()


SMPL_PATH = os.path.join("data", "body_models", "smpl", "SMPL_NEUTRAL.pkl")

smpl_model = SMPL(SMPL_PATH)

beta_tensor = torch.zeros(1, 10)
body_exp_tensor = torch.zeros(1, 10)

smpl_output = smpl_model(beta=beta_tensor)

# print(smpl_output)

verts = smpl_output.vertices[0].detach().numpy()
faces = smpl_model.faces

mesh = o3d.geometry.TriangleMesh()

mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.5, 0.5, 0.5])

# mesh = o3d.geometry.TriangleMesh.create_sphere()
# mesh.compute_vertex_normals()
o3d.visualization.draw(mesh)
