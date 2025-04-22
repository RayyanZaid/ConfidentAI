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

prompt = "Your task is to create a summary based on the information given in the following sentences about the interview. This information will be the tonality, seating posture, percent of eye contact maintained, and visual cues of the interviewee. The summary should be in a professional tone and no more than 250 words. Based on the content per behavior, determine whether it was good or ways they should improve that aspect" 
response = model.generate_content(prompt)

print("Response:\n")
print(response.text)


#step 1: download gcloud CLI via https://cloud.google.com/sdk/docs/install#windows
#step 2: sign in through terminal, select project, and set default region
#step 3: run gcloud 'auth application-default login' and it works