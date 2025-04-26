from summaryGeneration import generateResponseFromVideoInformation
from analyzeProsody import analyze_prosody
from oneVideo import compute_eye_contact_percent, CURRENT_GOOD_EYE_CNT_PCT
from A_startinterview import mockFacialGesture
import sys

def processVideo(video_path):
    """
    Process the video file to extract frames and perform analysis.
    """
    # Placeholder for video processing logic
    print(f"Processing video: {video_path}")



    facialGestureScore, facialGestureFeedback = mockFacialGesture(video_path)
    prosodyScore, prosodyFeedback = analyze_prosody(video_path)
    eyeContactFeedback = compute_eye_contact_percent(video_path)

    postureScore, postureFeedback = mockPostureAnalysis(video_path)

    # Combine feedback from all analyses
    
    # Concatenate feedback into a single string


    # This is the prompt for LLM

    # Final Score
    finalScore = (1/3) * prosodyScore + (1/3) * facialGestureScore + (1/3) * postureScore
    finalModelFeedback = (
       f"Your final score is: {finalScore*100}%\n\n\n"
        f"Facial Gesture Feedback: {facialGestureFeedback}\n\n\n"
        f"For Prosody Feedback, Keep this to 2 sentences, but mention the important stuff. You need to note the pitch, energy levels, and intensity. Don't sound robotic. If it's a high standard deviation, say it in human simple terms. Don't give the exact numbers, but say higher/lower. Here is the feedback: You scored {prosodyScore*100}% on this interview. Here is your feedback: {prosodyFeedback}\n\n\n\n"
        f"Eye Contact Feedback: If {eyeContactFeedback} is less than {CURRENT_GOOD_EYE_CNT_PCT}, then the interviee isn't maintaing eye contact enough. Give tips to improve eye contact such as being relaxed and confortable. That it's okay to glance away when speaking or thinking, but make sure to look at them when listening and making important points. Things in that nature. If the {eyeContactFeedback} is equal or more than the {CURRENT_GOOD_EYE_CNT_PCT}, then state that their eye contact maintence was good and no improvements are needed there. Make your response breif (2 sentences max) and natural, like a reviewer or teacher. \n\n\n\n"
        f"Posture Feedback: {postureFeedback}")
        



    # Generate summary of the feedback
    summary = generateResponseFromVideoInformation(finalModelFeedback)

    print("Summary of the interview:")
    print(summary)

    return summary

    



def mockFacialGesture(video_path):
    """
    Mock function to simulate facial expression analysis.
    """
    score = 1
    return score, "Facial expression was neutral"

def mockProsodyAnalysis(video_path):
    """
    Mock function to simulate prosody analysis.
    """
    return "Prosody was good"


def mockEyeContactAnalysis(video_path):
    """
    Mock function to simulate eye contact analysis.
    """
    return "Eye contact was maintained"


def mockPostureAnalysis(video_path):
    """
    Mock function to simulate posture analysis.
    """
    score = 1
    return score, "Posture was upright"
