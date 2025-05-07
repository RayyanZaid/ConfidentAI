import sys
import os
import subprocess
from pathlib import Path
try:
    from A_Final_inferencing_integration import analyze_facial_gestures
    DIRECT_IMPORT = True
except ImportError:
    DIRECT_IMPORT = False
DEFAULT_MODEL_PATH = "best_model_epoch_13.pt"
def mockFacialGesture(video_path, model_path=DEFAULT_MODEL_PATH):
    normalized_score = None
    if DIRECT_IMPORT:
        normalized_score = analyze_facial_gestures(video_path, model_path)
    else:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_dir, "A_Final_inferencing_integration.py")
            result = subprocess.run(
                [sys.executable, script_path, "--video_path", video_path, "--model_path", model_path],
                capture_output=True,
                text=True,
                check=True
            )
            for line in result.stdout.split('\n'):
                if line.startswith("normalized_score="):
                    score_str = line.split("=")[1].strip()
                    if score_str != "None":
                        normalized_score = float(score_str)
        except Exception as e:
            print(f"Error running facial gesture analysis: {e}")
    if normalized_score is None:
        return (
        None, "Unable to analyze facial gestures. Please check the video quality and ensure your face is visible.")
    if normalized_score >= 0.8:
        feedback = "Excellent facial expressions. Your expressions convey confidence and enthusiasm, making a very positive impression."
    elif normalized_score >= 0.5:
        feedback = "Good facial expressions (PASSING). Practice more varied and engaging facial expressions to strengthen your interview presence."
    else:
        feedback = "Facial expressions need improvement. Maintain consistent eye contact, especially during challenging questions, and show more enthusiasm during key moments."

    return (normalized_score, feedback)


def processVideo(video_path):
    print(f"Processing video: {video_path}")
    facial_score, facial_feedback = mockFacialGesture(video_path)
    if facial_score is not None:
        facialGestureFeedback = f"Score: {facial_score:.2f}\nFeedback: {facial_feedback}"
    else:
        facialGestureFeedback = f"Feedback: {facial_feedback}"
    finalModelFeedback = (
        f"Facial Gesture Feedback: {facialGestureFeedback}\n\n\n")
    print("Summary of the interview:")
    return finalModelFeedback


def mockPostureAnalysis(video_path):
    return "Posture was upright"

video_path = "/Users/sriramacharya/PycharmProjects/MIT_interview_dataset_cleaner/MIT_INTERVIEW_DATASET/Videos/P6.avi"
if __name__ == "__main__":
    video_path = video_path
    process = processVideo(video_path)
    print(process)
