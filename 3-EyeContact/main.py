import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

VIDEO_FOLDER = "3-EyeContact/MIT_DATA/Videos"
CSV_FILE = "3-EyeContact/MIT_DATA/results.csv"
PASS_THRESHOLD = 4.0
ATTENTION_REGION_RATIO = (0.45, 0.70)  

def is_eye_contact(iris_center, img_w, img_h, attention_region_ratio=ATTENTION_REGION_RATIO):
    region_w_ratio, region_h_ratio = attention_region_ratio

    x_min, y_max = 0, img_h
    x_max, y_min = img_w * region_w_ratio, img_h * (1 - region_h_ratio)

    x, y = iris_center
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)

results_df = pd.read_csv(CSV_FILE)
aggr_rows = results_df[results_df['Worker'] == 'AGGR']
pass_fail_dict = dict(zip(
    aggr_rows['Participant'].str.strip().str.lower(), 
    aggr_rows['RecommendHiring'] >= PASS_THRESHOLD)
)

mp_face_mesh = mp.solutions.face_mesh
data = []

for video_file in os.listdir(VIDEO_FOLDER):
    if video_file.endswith('.avi'):
        participant_id = os.path.splitext(video_file)[0].strip().lower()
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

                    if is_eye_contact(iris_center, w, h):
                        eye_contact_frames += 1

        cap.release()

        eye_contact_sec = eye_contact_frames / fps
        eye_contact_pct = (eye_contact_frames / total_frames) * 100
        passed = pass_fail_dict.get(participant_id, None)
        
        print(f"Processing participant: {participant_id}, Passed status: {passed}")

        if passed is not None:
            data.append({
                'Participant': participant_id,
                'EyeContactDuration': eye_contact_sec,
                'EyeContactPct': eye_contact_pct,
                'PassedInterview': passed
            })
        else:
            print(f"Participant ID '{participant_id}' not found in CSV. Skipping entry.")

final_df = pd.DataFrame(data)
final_df.to_csv('eye_contact_results.csv', index=False)

corr_duration, p_duration = pearsonr(final_df['EyeContactDuration'], final_df['PassedInterview'])
corr_pct, p_pct = pearsonr(final_df['EyeContactPct'], final_df['PassedInterview'])

print("Correlation Analysis:")
print(f"- Duration vs Pass: Corr={corr_duration:.3f}, P-value={p_duration:.3f}")
print(f"- Percent vs Pass: Corr={corr_pct:.3f}, P-value={p_pct:.3f}")

#test 1:
# Correlation Analysis ATTENTION_REGION_RATIO = (0.35, 0.65):
# - Duration vs Pass: Corr=0.090, P-value=0.291
# - Percent vs Pass: Corr=0.067, P-value=0.435

#test 2:
# Correlation Analysis:
# - Duration vs Pass: Corr=0.072, P-value=0.401
# - Percent vs Pass: Corr=0.011, P-value=0.901

#test3:
# Correlation Analysis:
# - Duration vs Pass: Corr=0.071, P-value=0.407
# - Percent vs Pass: Corr=0.047, P-value=0.584