import os
import pandas as pd
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

# Define the positions of the traffic lights. Replace with the actual values
traffic_lights_positions = [
    {"x_min": 762, "x_max": 788, "y_min": 330, "y_max": 390},  # Traffic light 1
    # No need to detect light 2, will copy result from light 1
    {"x_min": 1705, "x_max": 1735, "y_min": 282, "y_max": 362},  # Traffic light 3
    {"x_min": 1855, "x_max": 1880, "y_min": 290, "y_max": 355},  # Traffic light 4
]

# Define the brightness threshold below which a light is considered "shining"
SHINING_THRESHOLD = 50

# Function to get the color of a region in an image
def get_color(image, x_min, x_max, y_min, y_max):
    # Crop the image
    cropped_image = image[y_min:y_max, x_min:x_max]
    # Convert the image to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Divide the image into three sections
    section_height = (y_max - y_min) // 3
    sections = [thresholded[i*section_height:(i+1)*section_height, :] for i in range(3)]
    # Find the average brightness of each section
    averages = [np.mean(section) for section in sections]
    # If all sections are dim, return "shining"
    if all(average < SHINING_THRESHOLD for average in averages):
        return "shining"
    # Otherwise, return the color based on the brightest section
    return ["red", "yellow", "green"][averages.index(max(averages))]

# video_batches = [range(i, i + 10) for i in range(1, 6480, 10)]  # Put your actual video file names here
video_batches = [range(6481, 6489)]  # Put your actual video file names here
frame_counter = 0

# Initialize or load the output DataFrame
if os.path.exists("output.csv"):
    df_output = pd.read_csv("output.csv")
else:
    df_output = pd.DataFrame(columns=['frame', 'light_1', 'light_2', 'light_3', 'light_4'])

# Process each batch of videos
for video_batch in video_batches:
    for video_number in video_batch:
        clip = VideoFileClip(f"{video_number}.mp4")
        # Process each frame
        for i in range(int(clip.fps * clip.duration)):
            # Get the frame as a numpy array
            frame = clip.get_frame(i / clip.fps)
            # Get the colors of the traffic lights
            colors = [get_color(frame, **pos) for pos in traffic_lights_positions]
            # Copy the result from light 1 to light 2
            colors.insert(1, colors[0])
            # Append the result to the output DataFrame
            df_output = df_output.append({'frame': frame_counter, 'light_1': colors[0], 'light_2': colors[1], 'light_3': colors[2], 'light_4': colors[3]}, ignore_index=True)
            frame_counter += 1
            print(f"Frame {frame_counter} processed.")
        # Save the output DataFrame to a CSV file after each video
        df_output.to_csv("output.csv", index=False)

