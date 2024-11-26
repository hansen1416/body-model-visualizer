import time

import cv2
import open3d as o3d

# import numpy as np


class VideoPlayer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

    def play(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or cannot read the frame.")
                break

            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create an Open3D image from the frame
            image = o3d.geometry.Image(frame)

            # # Create a mesh or point cloud from the image (optional)
            # # Here, we simply display the image as a texture on a plane
            # width, height = frame.shape[1], frame.shape[0]
            # mesh = o3d.geometry.TriangleMesh.create_box(
            #     width=width, height=height, depth=0.01
            # )
            # mesh.compute_vertex_normals()
            # mesh.paint_uniform_color([1, 1, 1])  # Set the color to white

            # # Map the image as a texture
            # mesh.textures = [image]

            # Clear the visualizer and add the new mesh
            vis.clear_geometries()
            vis.add_geometry(image)

            # Update the visualizer
            vis.update_geometry(image)
            vis.poll_events()
            vis.update_renderer()

            # Control the speed of the animation
            time.sleep(0.03)

        self.cap.release()
        vis.destroy_window()


# Example usage
if __name__ == "__main__":

    import os

    video_dir = os.path.join(os.path.expanduser("~"), "Downloads", "videos")

    # iterate over `video_dir`
    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        player = VideoPlayer(video_path)
        player.play()

        break
