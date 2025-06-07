import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import threading
import time


class MeshAnimationApp:
    def __init__(self):
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Animated Mesh", 800, 600)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene_widget)

        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultLit"

        self.mesh = o3d.geometry.TriangleMesh.create_box()
        self.mesh.compute_vertex_normals()
        self.scene_widget.scene.add_geometry("mesh", self.mesh, self.material)

        bounds = self.mesh.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())

        self.frame = 0
        self.running = True
        threading.Thread(target=self.animate_mesh, daemon=True).start()

    def animate_mesh(self):
        while self.running:
            verts = np.asarray(self.mesh.vertices)
            # Example animation: sine wave deformation
            verts[:, 1] = np.sin(verts[:, 0] * 4 + self.frame * 0.1) * 0.1
            new_mesh = o3d.geometry.TriangleMesh()
            new_mesh.vertices = o3d.utility.Vector3dVector(verts)
            new_mesh.triangles = self.mesh.triangles
            new_mesh.compute_vertex_normals()

            def update_scene():
                self.scene_widget.scene.remove_geometry("mesh")
                self.scene_widget.scene.add_geometry("mesh", new_mesh, self.material)

            gui.Application.instance.post_to_main_thread(self.window, update_scene)
            self.frame += 1
            time.sleep(0.033)  # ~30 FPS

    def run(self):
        gui.Application.instance.run()
        self.running = False


MeshAnimationApp().run()
