# ConfidentAI

This is our backend infrastructure for a Mock Interviewer with Feedback. We measure 4 behaviors: 
1. Facial Gestures (Sriram) - Dataset: https://www.kaggle.com/code/drcapa/facial-expression-eda-cnn
2. Vocal Tone (Rayyan) - Dataset: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio/data
3. Eye Contact (Kartik) - https://github.com/rehg-lab/eye-contact-cnn
4. Posture (Sherwin) - https://github.com/itakurah/Sitting-Posture-Detection-YOLOv5%7C

We train Facial Gestures and Vocal Tone using datasets while using open-source projects Posture and Eye Contact.

At the end, we give the user personalized feedback based on these 4 metrics
