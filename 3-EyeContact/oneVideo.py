import cv2
import mediapipe as mp
import numpy as np

ATTENTION_REGION_RATIO = (0.45, 0.70)
CURRENT_GOOD_EYE_CNT_PCT = 57

def compute_eye_contact_percent(video_path):
    def is_eye_contact(iris_center, img_w, img_h, attention_region_ratio=ATTENTION_REGION_RATIO):
        region_w_ratio, region_h_ratio = attention_region_ratio
        x_min, y_max = 0, img_h
        x_max, y_min = img_w * region_w_ratio, img_h * (1 - region_h_ratio)
        x, y = iris_center
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"video path is wrong: {video_path}")
        return None

    eye_contact_frames = 0
    total_frames = 0

    mp_face_mesh = mp.solutions.face_mesh

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

    if total_frames > 0:
        eye_contact_pct = (eye_contact_frames / total_frames) * 100
        return eye_contact_pct
    else:
        return None