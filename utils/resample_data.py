import os
import numpy as np

input_dir = '../datasets/bukva_processed/processed_landmarks_xy'
output_dir = '../datasets/bukva_processed/processed_landmarks_xy_resampled'
os.makedirs(output_dir, exist_ok=True)

def resample_landmarks(data, target_frames):
    current_frames = data.shape[0]

    start_idx = min(2, current_frames - 1)
    indices = np.linspace(start_idx, current_frames - 1, target_frames, dtype=int)
    return data[indices]

# Process all files
for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        file_path = os.path.join(input_dir, filename)
        original_data = np.load(file_path)

        new_data = resample_landmarks(original_data, target_frames=24)

        np.save(os.path.join(output_dir, filename), new_data)

print("Done resampling all files!")