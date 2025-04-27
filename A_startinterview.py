import sys
import os
import subprocess
from pathlib import Path

# Import the facial gesture analysis function if available directly
try:
    from A_Final_inferencing_integration import analyze_facial_gestures

    DIRECT_IMPORT = True
except ImportError:
    DIRECT_IMPORT = False

# Default model path - update this to match your environment
DEFAULT_MODEL_PATH = "best_model_epoch_13.pt"
def mockFacialGesture(video_path, model_path=DEFAULT_MODEL_PATH):
    """
    Runs the inference helper and returns (score, feedback).

    Args:
        video_path: Path to the video file
        model_path: Path to the model checkpoint

    Returns:
        tuple: (score, feedback) where score is a normalized float between 0.0-1.0 
               and feedback is a two-sentence string
    """
    # Get normalized score (between 0.0 and 1.0)
    normalized_score = None

    if DIRECT_IMPORT:
        # Use direct function call if import worked
        normalized_score = analyze_facial_gestures(video_path, model_path)
    else:
        # Fall back to subprocess if module can't be imported
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_dir, "A_Final_inferencing_integration.py")

            result = subprocess.run(
                [sys.executable, script_path, "--video_path", video_path, "--model_path", model_path],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse the output
            for line in result.stdout.split('\n'):
                if line.startswith("normalized_score="):
                    score_str = line.split("=")[1].strip()
                    if score_str != "None":
                        normalized_score = float(score_str)
        except Exception as e:
            print(f"Error running facial gesture analysis: {e}")

    # Generate feedback based on the score
    if normalized_score is None:
        return (
        None, "Unable to analyze facial gestures. Please check the video quality and ensure your face is visible.")

    # Generate feedback based on the normalized score ranges
    if normalized_score >= 0.8:
        feedback = "Excellent facial expressions. Your expressions convey confidence and enthusiasm, making a very positive impression."
    elif normalized_score >= 0.5:
        feedback = "Good facial expressions (PASSING). Practice more varied and engaging facial expressions to strengthen your interview presence."
    else:
        feedback = "Facial expressions need improvement. Maintain consistent eye contact, especially during challenging questions, and show more enthusiasm during key moments."

    # Return the score and feedback as a tuple
    return (normalized_score, feedback)


def processVideo(video_path):
    """
    Process the video file to extract frames and perform analysis.
    """
    print(f"Processing video: {video_path}")

    # Get facial gesture score and feedback
    facial_score, facial_feedback = mockFacialGesture(video_path)

    # Format the facial gesture feedback for the final model input
    if facial_score is not None:
        facialGestureFeedback = f"Score: {facial_score:.2f}\nFeedback: {facial_feedback}"
    else:
        facialGestureFeedback = f"Feedback: {facial_feedback}"

    # Perform other analyses
#    prosodyScore, prosodyFeedback = analyze_prosody(video_path)
#    eyeContactFeedback = compute_eye_contact_percent(video_path)
#    postureFeedback = mockPostureAnalysis(video_path)

    # Combine feedback from all analyses
    finalModelFeedback = (
        f"Facial Gesture Feedback: {facialGestureFeedback}\n\n\n"
#        f"For Prosody Feedback, you need to note the pitch, energy levels, and intensity. Don't sound robotic. If it's a high standard deviation, say it in human simple terms. Here is the feedback: You scored {prosodyScore * 100}% on this interview. Here is your feedback: {prosodyFeedback}\n\n\n\n"
#        f"Eye Contact Feedback: If {eyeContactFeedback} is less than {CURRENT_GOOD_EYE_CNT_PCT}, then the interviee isn't maintaing eye contact enough. Give tips to improve eye contact such as being relaxed and confortable. That it's okay to glance away when speaking or thinking, but make sure to look at them when listening and making important points. Things in that nature. If the {eyeContactFeedback} is equal or more than the {CURRENT_GOOD_EYE_CNT_PCT}, then state that their eye contact maintence was good and no improvements are needed there. Make your response breif (2 sentences max) and natural, like a reviewer or teacher. \n\n\n\n"
#        f"Posture Feedback: {postureFeedback}"
    )

    # Generate summary of the feedback
#    summary = generateResponseFromVideoInformation(finalModelFeedback)

    print("Summary of the interview:")
 #   print(summary)

    return finalModelFeedback


def mockPostureAnalysis(video_path):
    """
    Mock function to simulate posture analysis.
    """
    return "Posture was upright"

video_path = "/Users/sriramacharya/PycharmProjects/MIT_interview_dataset_cleaner/MIT_INTERVIEW_DATASET/Videos/P6.avi"
# Main function for testing
if __name__ == "__main__":

    video_path = video_path
    process = processVideo(video_path)
    print(process)
