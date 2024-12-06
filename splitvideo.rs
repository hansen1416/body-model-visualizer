use ffmpeg_next::{
    format::input,
    media::Type,
    packet::Packet,
    software::scaling::context::Context as ScalingContext,
};
use std::fs::File;
use std::io::Write;

fn main() {
    // Load the video file
    let input_path = "path/to/your/video.mp4";
    let mut context = input(&input_path).unwrap();

    // Initialize variables for scene detection
    let mut last_frame: Option<Packet> = None;
    let mut scene_changes = Vec::new();
    let mut current_time = 0.0;

    // Iterate through the frames
    for (stream, packet) in context.packets() {
        if stream.kind() == Type::Video {
            // Get the current frame's timestamp
            let pts = packet.pts().unwrap_or(0);
            let frame_time = (pts as f64) / (stream.time_base().denominator as f64);

            // Detect scene change (simple thresholding)
            if let Some(last) = &last_frame {
                if is_scene_change(last, &packet) {
                    scene_changes.push(current_time);
                    println!("Scene change detected at: {:.2} seconds", frame_time);
                }
            }

            last_frame = Some(packet.clone());
            current_time = frame_time;
        }
    }

    // Add the end of the video as the last scene change
    scene_changes.push(current_time);

    // Splice the video based on detected scene changes
    splice_video(input_path, &scene_changes);
}

fn is_scene_change(last: &Packet, current: &Packet) -> bool {
    // Simple comparison based on packet size (you can implement a more sophisticated method)
    let last_size = last.data().len();
    let current_size = current.data().len();
    ((last_size as i32) - (current_size as i32)).abs() > 1000 // Threshold for scene change
}

fn splice_video(input_path: &str, scene_changes: &[f64]) {
    // Create output files for each scene
    for i in 0..scene_changes.len() - 1 {
        let start_time = scene_changes[i];
        let end_time = scene_changes[i + 1];
        let output_path = format!("scene_{}.mp4", i);

        // Use FFmpeg command to splice the video
        let command = format!(
            "ffmpeg -i {} -ss {:.2} -to {:.2} -c copy {}",
            input_path,
            start_time,
            end_time,
            output_path
        );

        // Execute the command
        let output = std::process::Command
            ::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .expect("Failed to execute FFmpeg command");

        if !output.status.success() {
            eprintln!("Error splicing video: {}", String::from_utf8_lossy(&output.stderr));
        } else {
            println!("Spliced scene {}: {} to {}", i, start_time, end_time);
        }
    }
}
