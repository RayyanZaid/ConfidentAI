import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

VIDEO_FOLDER = "3-EyeContact/MIT_DATA/Videos2"
CSV_FILE = "3-EyeContact/MIT_DATA/results2.csv"
OUTPUT_CSV = "eye_contact_results2.csv"
PASS_THRESHOLD = 0.6
CAMERA_REGION_RATIO = (0.3, 0.3) 

def is_eye_contact_camera(iris_center, img_w, img_h, camera_region_ratio=CAMERA_REGION_RATIO):
    region_w, region_h = camera_region_ratio

    x_center, y_center = img_w / 2, img_h / 2
    x_min = x_center - (region_w * img_w / 2)
    x_max = x_center + (region_w * img_w / 2)
    y_min = y_center - (region_h * img_h / 2)
    y_max = y_center + (region_h * img_h / 2)

    x, y = iris_center
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)

# Load evaluation data
results_df = pd.read_csv(CSV_FILE)
pass_fail_dict = dict(zip(
    results_df['video'].str.strip().str.lower(), 
    results_df['interview'] >= PASS_THRESHOLD)
)

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
output_data = []

for video_file in os.listdir(VIDEO_FOLDER):
    if video_file.endswith('.mp4'):
        video_id = video_file.strip().lower()  # Keep .mp4 extension because CSV has it
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        eye_contact_frames = 0
        total_frames = 0

        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                total_frames += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    h, w, _ = frame.shape

                    LEFT_IRIS = [474, 475, 476, 477]
                    RIGHT_IRIS = [469, 470, 471, 472]

                    def get_iris_center(indices):
                        coords = np.array([(landmarks.landmark[i].x * w,
                                            landmarks.landmark[i].y * h) for i in indices])
                        return np.mean(coords, axis=0)

                    iris_center = np.mean([
                        get_iris_center(LEFT_IRIS),
                        get_iris_center(RIGHT_IRIS)
                    ], axis=0)

                    if is_eye_contact_camera(iris_center, w, h):
                        eye_contact_frames += 1

        cap.release()

        eye_contact_sec = eye_contact_frames / fps
        eye_contact_pct = (eye_contact_frames / total_frames) * 100
        passed = pass_fail_dict.get(video_id, None)

        if passed is not None:
            output_data.append({
                'video': video_id,
                'EyeContactDuration': eye_contact_sec,
                'EyeContactPct': eye_contact_pct,
                'PassedInterview': passed
            })
            print(f"Processed {video_id}: Duration={eye_contact_sec:.2f}s, Percent={eye_contact_pct:.2f}%, Passed={passed}")
        else:
            print(f"Video '{video_id}' not found in CSV. Skipping entry.")

# Output results
final_df = pd.DataFrame(output_data)
final_df.to_csv(OUTPUT_CSV, index=False)

# Correlation analysis
corr_duration, p_duration = pearsonr(final_df['EyeContactDuration'], final_df['PassedInterview'])
corr_pct, p_pct = pearsonr(final_df['EyeContactPct'], final_df['PassedInterview'])

print("Correlation Analysis:")
print(f"- Duration vs Pass: Corr={corr_duration:.3f}, P-value={p_duration:.3f}")
print(f"- Percent vs Pass: Corr={corr_pct:.3f}, P-value={p_pct:.3f}")

# Correlation Analysis:
# - Duration vs Pass: Corr=-0.068, P-value=0.552
# - Percent vs Pass: Corr=-0.069, P-value=0.540