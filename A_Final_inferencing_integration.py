import argparse
import torch
import numpy as np
import os
import sys
from pathlib import Path

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Try different import approaches to handle various setup scenarios
try:
    # Try direct import first
    from facial_regression_inferencing_v2_with_features import (
        ModelConfig,
        FacialGestureRegressionModel,
        safe_load_checkpoint,
        process_video_for_inference
    )
except ImportError:
    try:
        # If that fails, try importing with module name
        import facial_regression_inferencing_v2_with_features as fr

        ModelConfig = fr.ModelConfig
        FacialGestureRegressionModel = fr.FacialGestureRegressionModel
        safe_load_checkpoint = fr.safe_load_checkpoint
        process_video_for_inference = fr.process_video_for_inference
    except ImportError:
        # As a last resort, try with the A_ prefix
        import A_facial_regression_inferencing_v2_with_features as fr

        ModelConfig = fr.ModelConfig
        FacialGestureRegressionModel = fr.FacialGestureRegressionModel
        safe_load_checkpoint = fr.safe_load_checkpoint
        process_video_for_inference = fr.process_video_for_inference

# Default model path
DEFAULT_MODEL_PATH = "/Users/sriramacharya/PycharmProjects/MIT_interview_dataset_cleaner/Extracted_Features/model_checkpoints/baseline_experiment_no_features/best_model_epoch_13.pt"

# Determine the appropriate device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")



def normalize_score(raw_score, original_min=0.0, original_max=10.0, target_min=0.0, target_max=1.0):
    # Handle edge cases
    if raw_score is None:
        return None

    # Simple linear normalization
    normalized = (raw_score - original_min) / (original_max - original_min)
    normalized = normalized * (target_max - target_min) + target_min

    # Ensure result stays within bounds
    normalized = max(target_min, min(target_max, normalized))

    return normalized


def initialize_facial_model(model_checkpoint_path):
    """
    Initialize the facial gesture regression model from a checkpoint.

    Args:
        model_checkpoint_path: Path to the model checkpoint

    Returns:
        tuple: (model, config, scaler) - Initialized model, configuration, and feature scaler
    """


    # Initialize configuration
    config = ModelConfig()

    try:
        # Load checkpoint
        checkpoint = safe_load_checkpoint(model_checkpoint_path, DEVICE)

        # Extract config from checkpoint if available
        if 'config' in checkpoint:
            for key, value in checkpoint['config'].items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Initialize model
        model = FacialGestureRegressionModel(config).to(DEVICE)

        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            raise KeyError("Checkpoint doesn't contain model weights under 'model_state_dict' or 'model' keys")

        # Initialize scaler
        checkpoint_dir = os.path.dirname(model_checkpoint_path)
        scaler_mean_path = os.path.join(checkpoint_dir, 'scaler_mean.npy')
        scaler_scale_path = os.path.join(checkpoint_dir, 'scaler_scale.npy')

        # Initialize scaler with training statistics if available
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
    """
    Analyze facial gestures in a video using the trained model.

    Args:
        video_path: Path to the video file
        model: Trained facial gesture regression model
        config: Model configuration
        scaler: Feature scaler

    Returns:
        dict: Analysis results including scores and feedback
    """
    try:
        # Process video and get predictions
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

        # Normalize the score to 0-1 range
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
    """
    Analyze facial gestures in a video and return a normalized score.

    Args:
        video_path: Path to the video file
        model_path: Path to the model checkpoint

    Returns:
        float: Normalized score (0.0-1.0) or None if analysis fails
    """
    # Initialize model
    model, config, scaler = initialize_facial_model(model_path)
    if model is None:

        return None

    # Perform analysis
    results = perform_facial_gesture_analysis(video_path, model, config, scaler)

    if not results["success"]:
        return None

    return results["normalized_score"]


def main():
    """
    Command-line interface for facial gesture analysis.
    Processes a video and returns a normalized score.
    """
    parser = argparse.ArgumentParser(description='Analyze facial gestures in interview videos')
    parser.add_argument('--video_path', required=True, help='Path to the video file')
    parser.add_argument('--model_path', required=True, help='Path to the model checkpoint')
    args = parser.parse_args()

    # Analyze facial gestures
    normalized_score = analyze_facial_gestures(args.video_path, args.model_path)

    # Print in a format that can be easily parsed by other scripts
    if normalized_score is not None:
        print(f"normalized_score={normalized_score:.4f}")
    else:
        print("normalized_score=None")


if __name__ == "__main__":
    main()