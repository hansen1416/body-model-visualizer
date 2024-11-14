import os

from smplx import create, SMPLX
from smplx.utils import SMPLXOutput
import open3d as o3d


model_folder = os.path.join("data", "body_models")
model_type = "smplx"
ext = "npz"
gender = "neutral"
use_face_contour = False
num_betas = 10
num_expression_coeffs = 10

model: SMPLX = create(
    model_folder,
    model_type=model_type,
    gender=gender,
    use_face_contour=use_face_contour,
    num_betas=num_betas,
    num_expression_coeffs=num_expression_coeffs,
    ext=ext,
)


output: SMPLXOutput = model.forward()


vertices = output.vertices.detach().cpu().numpy().squeeze()

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(model.faces)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.3, 0.3, 0.3])

geometry = [mesh]
# if plot_joints:
#     joints_pcl = o3d.geometry.PointCloud()
#     joints_pcl.points = o3d.utility.Vector3dVector(joints)
#     joints_pcl.paint_uniform_color([0.7, 0.3, 0.3])
#     geometry.append(joints_pcl)

o3d.visualization.draw_geometries(geometry)
