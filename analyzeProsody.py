import joblib
import librosa
import numpy as np

loaded_model = joblib.load("2-VocalTone-(Prosody)/DatasetRelatedFiles/models/regression_model.pkl")

# Feature Extraction Functions

# 1. Energy and Power
def extract_energy(speech):
    return librosa.feature.rms(y=speech)[0]  # Root mean square energy

# 2. Pitch (Fundamental Frequency) Statistics
def extract_pitch(speech, rate):
    pitch, _ = librosa.core.piptrack(y=speech, sr=rate)
    pitch = pitch[pitch > 0]  # Remove zero values
    return pitch

# 3. Pitch Statistics
def pitch_stats(pitch):
    min_pitch = np.min(pitch)
    max_pitch = np.max(pitch)
    mean_pitch = np.mean(pitch)
    pitch_sd = np.std(pitch)
    pitch_abs = np.mean(np.abs(pitch))
    pitch_quant = np.quantile(pitch, [0.25, 0.5, 0.75])
    diff_pitch_max_min = max_pitch - min_pitch
    diff_pitch_max_mean = max_pitch - mean_pitch
    diff_pitch_max_mode = max_pitch - np.median(pitch)
    return min_pitch, max_pitch, mean_pitch, pitch_sd, pitch_abs, pitch_quant, diff_pitch_max_min, diff_pitch_max_mean, diff_pitch_max_mode

# 4. Intensity (RMS)
def intensity_features(speech):
    intensity = librosa.feature.rms(y=speech)[0]
    intensity_min = np.min(intensity)
    intensity_max = np.max(intensity)
    intensity_mean = np.mean(intensity)
    intensity_sd = np.std(intensity)
    intensity_quant = np.quantile(intensity, [0.25, 0.5, 0.75])
    diff_int_max_min = intensity_max - intensity_min
    diff_int_max_mean = intensity_max - intensity_mean
    diff_int_max_mode = intensity_max - np.median(intensity)
    return intensity_min, intensity_max, intensity_mean, intensity_sd, intensity_quant, diff_int_max_min, diff_int_max_mean, diff_int_max_mode

# 5. Jitter and Shimmer
def jitter_shimmer(speech, rate):
    # Jitter: Measure of pitch variation
    pitch, voiced_flag = librosa.core.piptrack(y=speech, sr=rate)
    jitter = np.std(pitch[pitch > 0])
    
    # Shimmer: Measure of amplitude variation (calculated based on RMS)
    intensity = librosa.feature.rms(y=speech)[0]
    shimmer = np.std(intensity)
    
    return jitter, shimmer

# 6. Speech Rate and Pauses
def speech_rate(speech, rate):
    onset_env = librosa.onset.onset_strength(y=speech, sr=rate)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=rate)
    speak_rate = len(onset_frames) / (len(speech) / rate)  # Onsets per second
    return speak_rate

def pause_features(speech, rate):
    onset_env = librosa.onset.onset_strength(y=speech, sr=rate)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=rate)
    intervals = librosa.frames_to_time(onset_frames, sr=rate)  # Time intervals between onsets
    pauses = np.diff(intervals)  # Calculate pause durations
    max_pause = np.max(pauses) if len(pauses) > 0 else 0
    avg_pause = np.mean(pauses) if len(pauses) > 0 else 0
    total_pause_duration = np.sum(pauses)
    return max_pause, avg_pause, total_pause_duration

# 7. Rising and Falling Edges (Pitch changes)
def rising_falling_edges(pitch):
    rising = np.sum(np.diff(pitch) > 0)
    falling = np.sum(np.diff(pitch) < 0)
    max_rising = np.max(np.diff(pitch)[np.diff(pitch) > 0]) if len(np.diff(pitch)) > 0 else 0
    max_falling = np.min(np.diff(pitch)[np.diff(pitch) < 0]) if len(np.diff(pitch)) > 0 else 0
    avg_rise = np.mean(np.diff(pitch)[np.diff(pitch) > 0]) if len(np.diff(pitch)) > 0 else 0
    avg_fall = np.mean(np.diff(pitch)[np.diff(pitch) < 0]) if len(np.diff(pitch)) > 0 else 0
    return rising, falling, max_rising, max_falling, avg_rise, avg_fall

# 8. Loudness (RMS energy)
def loudness(speech):
    return librosa.feature.rms(y=speech)[0]


from moviepy import VideoFileClip
import tempfile
import librosa
import numpy as np

def analyze_prosody(video_path):

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
        audio_path = tmp_audio_file.name
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path)


    speech, rate = librosa.load(audio_path, sr=None)

    # Extract features
    energy = extract_energy(speech)
    pitch = extract_pitch(speech, rate)
    min_pitch, max_pitch, mean_pitch, pitch_sd, pitch_abs, pitch_quant, diff_pitch_max_min, diff_pitch_max_mean, diff_pitch_max_mode = pitch_stats(pitch)
    intensity_min, intensity_max, intensity_mean, intensity_sd, intensity_quant, diff_int_max_min, diff_int_max_mean, diff_int_max_mode = intensity_features(speech)
    jitter, shimmer = jitter_shimmer(speech, rate)
    speak_rate = speech_rate(speech, rate)
    max_pause, avg_pause, total_pause_duration = pause_features(speech, rate)

    # Create a feature vector
    feature_vector = np.array([
        np.mean(energy),
        min_pitch,
        max_pitch,
        mean_pitch,
        pitch_sd,
        intensity_min,
        intensity_max,
        intensity_mean,
        jitter,
        shimmer,
        speak_rate,
        max_pause,
        avg_pause
    ]).reshape(1, -1)

    # Predict score
    score = loaded_model.predict(feature_vector)[0]

    # Generate feedback
    feedback = generate_feedback(score, mean_pitch, intensity_mean, np.mean(energy))
    print(f"Prosody Score: {round(score * 7, 2)}/7")
    print(feedback)

    return score, feedback



# Feature Ranges for Good Scores: (from the fine_tuned model)
#                        mean        std
# Energy             0.009610   0.003262
# Pitch Min        152.356338   0.028201
# Pitch Max       3995.817406   0.343104
# Pitch Mean       989.273470  64.621149
# Pitch Std Dev    844.331216  45.801592
# Intensity Min      0.000956   0.000392
# Intensity Max      0.080463   0.034975
# Intensity Mean     0.009610   0.003262
# Jitter           844.331216  45.801592
# Shimmer            0.007931   0.002677
# Speech Rate        4.373822   1.043764
# Max Pause          3.212444   0.996461
# Avg Pause          0.246478   0.086813

# Feature Ranges for Bad Scores:
#                        mean         std
# Energy             0.007285    0.002695
# Pitch Min        152.353044    0.009132
# Pitch Max       3995.917206    0.173962
# Pitch Mean       930.038257  133.811968
# Pitch Std Dev    816.480155   90.327492
# Intensity Min      0.000966    0.000412
# Intensity Max      0.068384    0.033048
# Shimmer            0.006158    0.002599
# Speech Rate        4.081573    1.263227
# Max Pause          6.377412    9.732412
# Avg Pause          0.288427    0.172997


def generate_feedback(score, mean_pitch_sd, intensity_mean, energy):
    feedback = f"Score: {round(score * 7, 2)}/7 — "


    good_stats = {
        "mean_pitch_sd": {"mean": 844.33, "std": 45.80},
        "intensity_mean": {"mean": 0.00961, "std": 0.00326},
        "energy": {"mean": 0.00961, "std": 0.00326},
    }


    def check_feature(name, value):
        mean = good_stats[name]["mean"]
        std = good_stats[name]["std"]
        z = (value - mean) / std

        if abs(z) <= 1:
            return "good", f"{name.replace('_', ' ')} was ideal at {round(value, 5)}"
        elif z < -1:
            return "bad", f"{name.replace('_', ' ')} was too low ({round(value, 5)}), about {round(abs(z), 2)} std below expected"
        else:
            return "bad", f"{name.replace('_', ' ')} was too high ({round(value, 5)}), about {round(z, 2)} std above expected"


    pitch_status, pitch_msg = check_feature("mean_pitch_sd", mean_pitch_sd)
    intensity_status, intensity_msg = check_feature("intensity_mean", intensity_mean)
    energy_status, energy_msg = check_feature("energy", energy)

    good_parts = []
    bad_parts = []

    if pitch_status == "good":
        good_parts.append(pitch_msg)
    else:
        bad_parts.append(pitch_msg)

    if intensity_status == "good":
        good_parts.append(intensity_msg)
    else:
        bad_parts.append(intensity_msg)

    if energy_status == "good":
        good_parts.append(energy_msg)
    else:
        bad_parts.append(energy_msg)

    # Build feedback paragraph
    if score >= (4 / 7):
        feedback += "Good prosody overall! "
        if good_parts:
            feedback += "Strengths: " + "; ".join(good_parts) + ". "
        if bad_parts:
            feedback += "Consider improving: " + "; ".join(bad_parts) + "."
    else:
        feedback += "Needs improvement. "
        if bad_parts:
            feedback += "Focus on: " + "; ".join(bad_parts) + ". "
        if good_parts:
            feedback += "However, you did well on: " + "; ".join(good_parts) + "."

    return feedback



    
if __name__ == "__main__":
    score, feedback = analyze_prosody("MIT_INTERVIEW_DATASET/Videos/P1.avi")

    print(f"Prosody Score: {score}")
    print(f"Feedback: {feedback}")
