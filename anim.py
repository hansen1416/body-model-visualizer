import open3d as o3d
import numpy as np
import time

# Create a mesh
mesh = o3d.geometry.TriangleMesh()

# Example vertices and faces
verts = np.random.rand(100, 3)  # Random vertices
faces = np.random.randint(0, 100, (200, 3))  # Random faces
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.5, 0.5, 0.5])

# Create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the mesh to the visualizer
vis.add_geometry(mesh)

# Animation loop
for i in range(100):  # Number of frames
    # Update the vertices (for example, create a wave effect)
    verts = np.asarray(mesh.vertices)
    verts[:, 1] = np.sin(
        verts[:, 0] + i * 0.1
    )  # Update y-coordinates based on a sine wave
    mesh.vertices = o3d.utility.Vector3dVector(verts)

    # Update the mesh in the visualizer
    vis.update_geometry(mesh)

    # Optionally, you can update the visualizer's view
    vis.poll_events()
    vis.update_renderer()

    # Control the speed of the animation
    time.sleep(0.1)

# Close the visualizer
vis.destroy_window()
