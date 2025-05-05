

# check if API works via:
import time
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from vertexai.batch_prediction import BatchPredictionJob
from google.cloud import storage
import os
import json
from vertexai.generative_models import GenerativeModel, Part
import csv

PROJECT_ID = "infinite-chain-456905-k1"
vertexai.init(project=PROJECT_ID, location="us-central1")
model = GenerativeModel("gemini-1.5-pro-002")

def generateResponseFromVideoInformation(videoInformation):
    prompt = f"""Your task is to create a summary based on the information given in the following sentences about the interview. 
    This information will be the tonality, seating posture, percent of eye contact maintained, and visual cues of the interviewee. 
    The summary should be in a professional tone and no more than 250 words. 
    Based on the content per behavior, determine whether it was good or ways they should improve that aspect. Here is the information: {videoInformation}. 
        "Return the feedback above in a json format like this:
        "Final Score": "",
        "Facial Gesture Feedback": "",
        "Prosody Feedback": "",
        "Eye Contact Feedback": "",
        "Posture Feedback": ""
        
    """ 

    response = model.generate_content(prompt)

    print("Response:\n")
    print(response.text)

    return response.text

#Baseline of just Gemini
# def generateResponseFromVideoInformation(videoInformation):
#     prompt = f"""Your task is to create a short summary based on analyzing 4 factors about the interview. 
#     These factors are tonality, seating posture, percent of eye contact maintained, and visual cues of the interviewee. 
#     For Facial Gesture Feedback, you should analyze the interviewee's facial expressions and gestures during the interview.
#     For Prosody Feedback, you should analyze the interviewee's tone of voice, pitch, and energy levels.
#     For Eye Contact Feedback, you should analyze the interviewee's eye contact with the interviewer.
#     For Posture Feedback, you should analyze the interviewee's body language and posture during the interview.
#     Determine whether the interviewee's performance in each of these areas was good or if they need to improve.
#     The summary should be in a professional tone and no more than 250 words. 
#     Based on the content per behavior, determine whether it was good or ways they should improve that aspect. Here is the information: {videoInformation}. 
#         "Return the feedback above in a json format like this:
#         "Final Score": "",
#         "Facial Gesture Feedback": "",
#         "Prosody Feedback": "",
#         "Eye Contact Feedback": "",
#         "Posture Feedback": ""
        
#     """ 

#     response = model.generate_content(prompt)

#     print("Response:\n")
#     print(response.text)

#     return response.text

#step 1: download gcloud CLI via https://cloud.google.com/sdk/docs/install#windows
#step 2: sign in through terminal, select project, and set default region
#step 3: run gcloud 'auth application-default login' and it works



# test

if __name__ == "__main__":
    # Example video information
    videoInformation = {
        "tonality": "The interviewee's tonality was confident and assertive, with a steady pace and clear articulation.",
        "seating_posture": "The interviewee maintained an upright seating posture, which conveyed attentiveness and engagement.",
        "eye_contact": "The interviewee maintained eye contact approximately 70% of the time, indicating confidence and connection with the interviewer.",
        "visual_cues": "The interviewee used appropriate hand gestures to emphasize points, but occasionally fidgeted with a pen."
    }

    # Generate summary
    summary = generateResponseFromVideoInformation(videoInformation)
    print("Generated Summary:\n", summary)