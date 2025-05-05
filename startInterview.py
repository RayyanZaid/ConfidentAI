from summaryGeneration import generateResponseFromVideoInformation
from analyzeProsody import analyze_prosody
from oneVideo import compute_eye_contact_percent, CURRENT_GOOD_EYE_CNT_PCT
from A_startinterview import mockFacialGesture
import sys
import cv2  
import mediapipe as mp  
import numpy as np

def processVideo(video_path):
    """
    Process the video file to extract frames and perform analysis.
    """
    # Placeholder for video processing logic
    print(f"Processing video: {video_path}")



    facialGestureScore, facialGestureFeedback = mockFacialGesture(video_path)
    prosodyScore, prosodyFeedback = analyze_prosody(video_path)
    eyeContactFeedback = compute_eye_contact_percent(video_path)

    postureScore, postureFeedback = posture(video_path)

    # Combine feedback from all analyses
    
    # Concatenate feedback into a single string


    # This is the prompt for LLM

    # Final Score
    finalScore = (1/3) * prosodyScore + (1/3) * facialGestureScore + (1/3) * postureScore
    finalModelFeedback = (
       f"Your final score is: {finalScore*100}%\n\n\n"
        f"Facial Gesture Feedback: {facialGestureFeedback}\n\n\n"
        f"For Prosody Feedback, Keep this to 2 sentences, but mention the important stuff. You need to note the pitch, energy levels, and intensity. Don't sound robotic. If it's a high standard deviation, say it in human simple terms. Don't give the exact numbers, but say higher/lower. Here is the feedback: You scored {prosodyScore*100}% on this interview. Here is your feedback: {prosodyFeedback}\n\n\n\n"
        f"Eye Contact Feedback: If {eyeContactFeedback} is within 15 of {CURRENT_GOOD_EYE_CNT_PCT}, then the interviee is maintaing a good level of eye contact. Depending on whether they are too far above or below the range, give tips accordingly. To improve eye contact say things like to be more relaxed and comfortable. That it's okay to glance away when speaking or thinking, but make sure to look at them when listening and making important points. Things in that nature. If the {eyeContactFeedback} is within 15 of {CURRENT_GOOD_EYE_CNT_PCT}, then state that their eye contact maintence was good and no improvements are needed there. Make your response brief (2 sentences max) and natural, like a reviewer or teacher. \n\n\n\n"
        f"Posture Feedback: If {postureScore} is lower than 0.6 than the interviee doesn't have good posture. Keep your analysis to 2 sentences. Here is some context feedback that could help with your analysis {postureFeedback}. All the scores here are normalized between 0 and 1 for different metrics of posture.")
        



    # Generate summary of the feedback
    summary = generateResponseFromVideoInformation(finalModelFeedback)

    print("Summary of the interview:")
    print(summary)

    return summary

    

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


def posture(video_path):
    """
    Mock function to simulate posture analysis.
    """
    def get_feedback(score, metric_name): #just general feedback
        level = int(score * 10)
        if level <= 1:
            return f"{metric_name}: Severe issue detected. Needs immediate correction."
        elif level <= 3:
            return f"{metric_name}: Poor posture. Focus on improving this area."
        elif level <= 5:
            return f"{metric_name}: Needs improvement."
        elif level <= 7:
            return f"{metric_name}: Doing okay, but there’s room for improvement."
        elif level <= 8:
            return f"{metric_name}: Good posture. Stay consistent."
        elif level <= 9:
            return f"{metric_name}: Great job! Very stable."
        else:
            return f"{metric_name}: Excellent! Keep up the perfect posture."


    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()  # initialize pose detector
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    
    frame_counter = 0
    total_head_tilt_score = 0
    total_scrunch_score = 0
    total_leaning_score = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
    
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
    
            # 2D coordinates
            left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
            right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
            nose = np.array([landmarks[mp_pose.PoseLandmark.NOSE].x,
                             landmarks[mp_pose.PoseLandmark.NOSE].y])
    
            # 3D coordinates for considering leaning posture
            left_shoulder_3d = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z])
            right_shoulder_3d = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z])
            nose_3d = np.array([landmarks[mp_pose.PoseLandmark.NOSE].x,
                                landmarks[mp_pose.PoseLandmark.NOSE].y,
                                landmarks[mp_pose.PoseLandmark.NOSE].z])
    
    
            neck_base = (left_shoulder + right_shoulder) / 2 # calc angle of head tilt via x and y axis
            dx = nose[0] - neck_base[0]
            dy = nose[1] - neck_base[1]
            head_tilt_angle = np.degrees(np.arctan2(dy, dx))
            head_tilt_score = max(0, abs(head_tilt_angle) / 220)
    
            v1 = left_shoulder - neck_base
            v2 = right_shoulder - neck_base
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            angle = np.arccos(dot_product / (norm_v1 * norm_v2)) * (180 / np.pi)
            shoulder_scrunch_score = max(0, 1 - ((180 - angle) / 145)) # create two vector to do dot product
    
            shoulders_z = (left_shoulder_3d[2] + right_shoulder_3d[2]) / 2 # in z axis consider difference
            nose_z = nose_3d[2]
            z_diff = shoulders_z - nose_z  
            leaning_score = max(0, min(1, z_diff / 0.4))  
    
           
            frame_counter += 1
            total_head_tilt_score += head_tilt_score
            total_scrunch_score += shoulder_scrunch_score
            total_leaning_score += leaning_score
            frame_count += 1
    
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
        cv2.imshow("Posture Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    if frame_count > 0:
        final_head_tilt_score = total_head_tilt_score / frame_count  #get the scores
        final_scrunch_score = total_scrunch_score / frame_count
        final_leaning_score = total_leaning_score / frame_count
        final_posture_score = (final_head_tilt_score + final_scrunch_score + final_leaning_score) / 3
    
        feedback_head = get_feedback(final_head_tilt_score, "Head Tilt")
        feedback_scrunch = get_feedback(final_scrunch_score, "Shoulder Scrunch")
        feedback_lean = get_feedback(final_leaning_score, "Leaning Posture")
        
        feedback_text = (
        f"Final Posture Score (normalized between 0 and 1): {final_posture_score:.2f}\n"
        "Posture Feedback Summary:\n"
        f"Head Tilt ({final_head_tilt_score:.2f}): {feedback_head}\n"
        f"Shoulder Scrunch ({final_scrunch_score:.2f}): {feedback_scrunch}\n"
        f"Leaning Posture ({final_leaning_score:.2f}): {feedback_lean}"
        )
    
    print(feedback_text)
    
    cap.release()
    cv2.destroyAllWindows()
    return final_posture_score, feedback_text
