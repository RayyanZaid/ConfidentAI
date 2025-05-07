import argparse
import torch
import numpy as np
import os
import sys
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
try:
    from A_facial_regression_inferencing_v2_with_features import (
        ModelConfig,
        FacialGestureRegressionModel,
        safe_load_checkpoint,
        process_video_for_inference
    )
except ImportError:
    try:
        import facial_regression_inferencing_v2_with_features as fr

        ModelConfig = fr.ModelConfig
        FacialGestureRegressionModel = fr.FacialGestureRegressionModel
        safe_load_checkpoint = fr.safe_load_checkpoint
        process_video_for_inference = fr.process_video_for_inference
    except ImportError:
        import A_facial_regression_inferencing_v2_with_features as fr

        ModelConfig = fr.ModelConfig
        FacialGestureRegressionModel = fr.FacialGestureRegressionModel
        safe_load_checkpoint = fr.safe_load_checkpoint
        process_video_for_inference = fr.process_video_for_inference

DEFAULT_MODEL_PATH = "best_model_epoch_13.pt"
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")



def normalize_score(raw_score, original_min=0.0, original_max=10.0, target_min=0.0, target_max=1.0):
    if raw_score is None:
        return None

    normalized = (raw_score - original_min) / (original_max - original_min)
    normalized = normalized * (target_max - target_min) + target_min

    normalized = max(target_min, min(target_max, normalized))

    return normalized


def initialize_facial_model(model_checkpoint_path):
    config = ModelConfig()

    try:
        checkpoint = safe_load_checkpoint(model_checkpoint_path, DEVICE)

        if 'config' in checkpoint:
            for key, value in checkpoint['config'].items():
                if hasattr(config, key):
                    setattr(config, key, value)

        model = FacialGestureRegressionModel(config).to(DEVICE)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            raise KeyError("Checkpoint doesn't contain model weights under 'model_state_dict' or 'model' keys")

        checkpoint_dir = os.path.dirname(model_checkpoint_path)
        scaler_mean_path = os.path.join(checkpoint_dir, 'scaler_mean.npy')
        scaler_scale_path = os.path.join(checkpoint_dir, 'scaler_scale.npy')
        inference_scaler = None
        if os.path.exists(scaler_mean_path) and os.path.exists(scaler_scale_path):
            from sklearn.preprocessing import StandardScaler
            inference_scaler = StandardScaler()
            inference_scaler.mean_ = np.load(scaler_mean_path)
            inference_scaler.scale_ = np.load(scaler_scale_path)

        else:
            None

        return model, config, inference_scaler

    except Exception as e:

        return None, None, None

def perform_facial_gesture_analysis(video_path, model, config, scaler):
    try:
        prediction, _ = process_video_for_inference(
            video_path, model, scaler, config, DEVICE
        )

        if prediction is None:
            return {
                "success": False,
                "score": None,
                "normalized_score": None,
                "error": "Failed to detect faces in the video"
            }

        normalized_score = normalize_score(prediction)

        return {
            "success": True,
            "score": float(prediction),
            "normalized_score": float(normalized_score)
        }

    except Exception as e:

        return {
            "success": False,
            "score": None,
            "normalized_score": None,
            "error": str(e)
        }

def analyze_facial_gestures(video_path, model_path=DEFAULT_MODEL_PATH):
    model, config, scaler = initialize_facial_model(model_path)
    if model is None:

        return None
    results = perform_facial_gesture_analysis(video_path, model, config, scaler)
    if not results["success"]:
        return None
    return results["normalized_score"]

def main():
    parser = argparse.ArgumentParser(description='Analyze facial gestures in interview videos')
    parser.add_argument('--video_path', required=True, help='Path to the video file')
    parser.add_argument('--model_path', required=True, help='Path to the model checkpoint')
    args = parser.parse_args()
    normalized_score = analyze_facial_gestures(args.video_path, args.model_path)
    if normalized_score is not None:
        print(f"normalized_score={normalized_score:.4f}")
    else:
        print("normalized_score=None")


if __name__ == "__main__":
    main()
