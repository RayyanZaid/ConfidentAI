import os
import re
import math
import json
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import mediapipe as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GraphConv
try:
    from torch.serialization import safe_globals, add_safe_globals
    from contextlib import nullcontext
    import numpy as _np
    safe_types = [
        _np.dtype, _np.ndarray, _np.generic,
        _np.int8, _np.int16, _np.int32, _np.int64,
        _np.uint8, _np.uint16, _np.uint32, _np.uint64,
        _np.float16, _np.float32, _np.float64,
        _np.complex64, _np.complex128, _np.bool_,
        _np.core.multiarray.scalar,  
        _np.dtypes.Float64DType  
    ]

    try:
        add_safe_globals(safe_types)
        for attr_name in dir(_np.dtypes):
            if attr_name.endswith('DType'):
                try:
                    dtype_class = getattr(_np.dtypes, attr_name)
                    add_safe_globals([dtype_class])
                except:
                    pass
    except Exception as e:
        print(f"Warning: Could not add all safe types: {e}")

    safe_globals_cm = safe_globals(safe_types)
except ImportError:
    from contextlib import nullcontext

    safe_globals_cm = nullcontext()


def safe_load_checkpoint(checkpoint_path, device=None):
    print(f"Loading checkpoint from {checkpoint_path}")
    try:
        with safe_globals_cm:
            return torch.load(checkpoint_path, map_location=device) if device else torch.load(checkpoint_path)
    except Exception as e:
        print(f"First load attempt failed with error: {e}")
        try:
            print("Attempting to load with weights_only=False (less secure)")
            return torch.load(checkpoint_path, map_location=device, weights_only=False) if device else torch.load(
                checkpoint_path, weights_only=False)
        except Exception as e2:
            print(f"Second load attempt failed with error: {e2}")
            try:
                if hasattr(torch.serialization, 'add_safe_globals'):
                    import numpy as np
                    print("Adding numpy.core.multiarray.scalar to safe globals")
                    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                    for attr_name in dir(np.dtypes):
                        if attr_name.endswith('DType'):
                            try:
                                dtype_class = getattr(np.dtypes, attr_name)
                                torch.serialization.add_safe_globals([dtype_class])
                            except:
                                pass
                    return torch.load(checkpoint_path, map_location=device) if device else torch.load(checkpoint_path)
            except Exception as e3:
                pass
            raise RuntimeError(f"Failed to load checkpoint: {e2}")

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS device for acceleration")
else:
    DEVICE = torch.device("cpu")
    print("MPS not available, using CPU")
class ModelConfig:
    """Configuration for architecture and training"""

    def __init__(self):
        self.num_landmarks = 468
        self.landmark_dim = 3
        self.feature_dim = self.num_landmarks * self.landmark_dim
        self.gcn_hidden_dim = 128
        self.gcn_layers = 3
        self.temporal_kernel_size = 3
        self.transformer_dim = 256
        self.num_heads = 4
        self.num_transformer_layers = 2
        self.transformer_feedforward_dim = 512
        self.transformer_dropout = 0.1
        self.regression_hidden_dims = [256, 128, 64]
        self.output_dim = 1
        self.batch_size = 16
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.max_epochs = 100
        self.early_stopping_patience = 10
        self.validation_split = 0.2
        self.target_label = "Overall"
        self.facial_features = ['EyeContact', 'Smiled', 'Engaged', 'Excited', 'Friendly', 'EngagingTone', 'Authentic',
                                'NotAwkward']
        self.facial_mode = False
def create_facial_landmark_graph(num_landmarks=468):
    edges = []
    for i in range(17): edges.append((i, i + 1))
    for i in range(17, 22): edges.append((i, i + 1))
    for i in range(22, 27): edges.append((i, i + 1))
    for i in range(27, 30): edges.append((i, i + 1))
    for i in range(31, 36): edges.append((i, i + 1))
    for i in range(36, 41): edges.append((i, i + 1))
    edges.append((41, 36))
    for i in range(42, 47): edges.append((i, i + 1))
    edges.append((47, 42))
    for i in range(48, 59): edges.append((i, i + 1))
    edges.append((59, 48))
    for i in range(60, 67): edges.append((i, i + 1))
    edges.append((67, 60))
    for i in range(68, num_landmarks):
        if i < num_landmarks - 1: edges.append((i, i + 1))
        if i < num_landmarks - 5: edges.append((i, i + 5))
    idx = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([idx, idx.flip(0)], dim=1)
class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        self.gcn = GraphConv(in_c, out_c)
        pad = (kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm1d(out_c), nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size, padding=pad),
            nn.BatchNorm1d(out_c), nn.ReLU())
        self.res = nn.Identity() if in_c == out_c else nn.Conv1d(in_c, out_c, 1)
        self.residual = self.res  

    def forward(self, x, edge_index, num_landmarks=468):
        res = x
        x = self.gcn(x, edge_index)
        bs = x.size(0) // num_landmarks
        x = x.view(bs, num_landmarks, -1).permute(0, 2, 1)
        x = self.tcn(x)
        r = res.view(bs, num_landmarks, -1).permute(0, 2, 1)
        r = self.res(r)
        x = x + r
        x = x.permute(0, 2, 1).reshape(-1, x.size(1))
        return x
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.drop = nn.Dropout(dropout)

    def forward(self, x): return self.drop(x + self.pe[:, :x.size(1)])
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hid, heads, layers, ff_dim, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hid)
        enc_layer = nn.TransformerEncoderLayer(d_model=hid, nhead=heads, dim_feedforward=ff_dim,
                                               dropout=dropout, batch_first=True, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.output_projection = nn.Linear(hid, input_dim)
        self.pos = PositionalEncoding(hid, dropout)
        self.pos_encoder = self.pos 

    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = self.pos(x)
        x = self.transformer_encoder(x, mask)
        return self.output_projection(x)


class AttentiveRegressionHead(nn.Module):
    def __init__(self, in_dim, h_dims, out_dim=1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in h_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.attn = nn.Linear(h_dims[-1], 1)
        self.attention = self.attn 
        self.output_projection = nn.Linear(h_dims[-1], out_dim)
        self.output_proj = self.output_projection 
        self.out = self.output_projection  

    def forward(self, x):
        bs, sl, _ = x.shape
        x = x.view(bs * sl, -1)
        x = self.mlp(x)
        x = x.view(bs, sl, -1)
        w = F.softmax(self.attn(x), dim=1)
        feat = torch.sum(w * x, dim=1)
        return self.out(feat), w


class FacialGestureRegressionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer('edge_index', create_facial_landmark_graph(cfg.num_landmarks))
        in_c = cfg.landmark_dim
        self.stgcn_layers = nn.ModuleList([STGCNBlock(in_c if i == 0 else cfg.gcn_hidden_dim,
                                                      cfg.gcn_hidden_dim,
                                                      kernel_size=cfg.temporal_kernel_size)
                                           for i in range(cfg.gcn_layers)])
        self.transformer = TransformerEncoder(cfg.gcn_hidden_dim, cfg.transformer_dim,
                                              cfg.num_heads, cfg.num_transformer_layers,
                                              cfg.transformer_feedforward_dim, cfg.transformer_dropout)
        self.regression_head = AttentiveRegressionHead(cfg.gcn_hidden_dim, cfg.regression_hidden_dims, cfg.output_dim)

    def forward(self, x):
        bs, nl, ld = x.shape
        x = x.reshape(-1, ld)
        for m in self.stgcn_layers:
            x = m(x, self.edge_index, nl)
        x = x.reshape(bs, nl, -1)
        x = self.transformer(x)
        return self.regression_head(x)

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom state dict loading with key mapping for backward compatibility.

        This method handles the mapping between old model checkpoint keys and
        the current model architecture keys.
        """
        key_mapping = {
            r'stgcn_layers.(\d+).residual': r'stgcn_layers.\1.res',
            r'regression_head.attention': r'regression_head.attn',
            r'regression_head.output_proj': r'regression_head.output_projection',
        }

        new_state_dict = {}
        for key, value in state_dict.items():
            mapped_key = key
            for old_pattern, new_pattern in key_mapping.items():
                mapped_key = re.sub(old_pattern, new_pattern, mapped_key)
            new_state_dict[mapped_key] = value

        if 'transformer.pos_encoder.pe' in new_state_dict and not hasattr(self.transformer, 'pos_encoder'):
            del new_state_dict['transformer.pos_encoder.pe']
        return super().load_state_dict(new_state_dict, strict=False)


def load_turker_scores(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded turker scores from {csv_path}")
        print(f"Found {len(df)} rows and {len(df.columns)} columns")
        if 'Participant' in df.columns:
            df['Participant'] = df['Participant'].str.lower()
            print("Converted participant IDs to lowercase for matching")

        return df
    except Exception as e:
        print(f"Error loading turker scores: {e}")
        return None


def extract_participant_id(video_path):
    filename = os.path.basename(video_path)
    filename_no_ext = os.path.splitext(filename)[0]

    return filename_no_ext, filename_no_ext.lower()

def get_ground_truth(turker_df, participant_id, participant_id_lower, target_label):
    if turker_df is None:
        return None

    filtered = turker_df[(turker_df['Participant'] == participant_id_lower) &
                         (turker_df['Worker'] == 'AGGR')]

    if len(filtered) == 0:
        filtered = turker_df[turker_df['Participant'] == participant_id_lower]
        if len(filtered) == 0:
            print(f"No data found for participant {participant_id} (looked for {participant_id_lower} in CSV)")
            return None

        if target_label in filtered.columns:
            mean_score = filtered[target_label].mean()
            print(f"Using mean score from {len(filtered)} workers for {participant_id}")
            return mean_score
        else:
            print(f"Target label '{target_label}' not found in columns")
            return None
    if target_label in filtered.columns:
        return filtered[target_label].values[0]
    else:
        print(f"Target label '{target_label}' not found in columns")
        return None


def process_video_for_inference(video_path, model, scaler, config, device):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_interval = int(fps)
    num_samples = int(frame_count / sample_interval)

    all_landmarks = []

    for frame_idx in tqdm(range(0, frame_count, sample_interval), desc="Processing video frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            landmarks = np.array([
                [landmark.x, landmark.y, landmark.z]
                for landmark in face_landmarks.landmark
            ])

            all_landmarks.append(landmarks)

    cap.release()
    face_mesh.close()

    if not all_landmarks:
        print(f"No faces detected in video {video_path}")
        return None, None

    mean_landmarks = np.mean(all_landmarks, axis=0)

    landmark_shape = mean_landmarks.shape
    print(f"Detected landmarks shape: {landmark_shape}")
    print(f"Expected landmarks shape: ({config.num_landmarks}, {config.landmark_dim})")

    if landmark_shape[0] != config.num_landmarks:
        print(f"Warning: Detected {landmark_shape[0]} landmarks but model expects {config.num_landmarks}")
        if landmark_shape[0] > config.num_landmarks:
            mean_landmarks = mean_landmarks[:config.num_landmarks]
        else:
            padded_landmarks = np.zeros((config.num_landmarks, landmark_shape[1]))
            padded_landmarks[:landmark_shape[0]] = mean_landmarks
            mean_landmarks = padded_landmarks

    flattened = mean_landmarks.flatten().reshape(1, -1)
    if scaler is not None:
        if flattened.shape[1] == len(scaler.mean_):
            scaled_features = scaler.transform(flattened)
            print("Applied training scaler normalization")
        else:
            print(f"Warning: Feature dimension mismatch. Expected {len(scaler.mean_)}, got {flattened.shape[1]}")
            scaled_features = (flattened - np.mean(flattened, axis=1, keepdims=True)) / (
                        np.std(flattened, axis=1, keepdims=True) + 1e-8)
            print(f"Applied fallback normalization")
    else:
        print("Using fallback normalization")
        scaled_features = (flattened - np.mean(flattened, axis=1, keepdims=True)) / (
                    np.std(flattened, axis=1, keepdims=True) + 1e-8)
    features = torch.FloatTensor(scaled_features).reshape(
        1, config.num_landmarks, config.landmark_dim).to(device)

    model.eval()
    with torch.no_grad():
        prediction, attention_weights = model(features)

    return prediction.item(), attention_weights.cpu().numpy()


def visualize_attention(attention_weights, landmarks=None, save_path=None):
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 1, 1)
    plt.imshow(attention_weights.reshape(-1, 1), cmap='hot', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title('Facial Landmark Attention Weights')
    plt.xlabel('Landmark Index')
    plt.ylabel('Attention Intensity')

    if save_path:
        plt.savefig(save_path)
        print(f"Attention visualization saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run inference with a trained facial gesture regression model')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory with saved model or path to specific checkpoint')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to video for inference')
    parser.add_argument('--target_label', default='Overall',
                        help='Target label to predict')
    parser.add_argument('--facial_mode', action='store_true',
                        help='Use facial mode for prediction')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize attention weights')
    parser.add_argument('--turker_scores_path', type=str,
                        default=None,
                        help='Path to turker scores CSV file for ground truth comparison')
    args = parser.parse_args()
    config = ModelConfig()
    config.target_label = args.target_label
    config.facial_mode = args.facial_mode

    print(f"Starting inference mode with video: {args.video_path}")
    if os.path.isfile(args.model_dir) and args.model_dir.endswith('.pt'):
        checkpoint_path = args.model_dir
    else:
        model_dir_path = Path(args.model_dir)
        ckpts = []
        for pattern in ['best_*.pt', 'best_model_epoch_*.pt']:
            ckpts.extend(model_dir_path.glob(pattern))

        if not ckpts:
            raise FileNotFoundError(f"No checkpoint files found in directory {args.model_dir}")
        checkpoint_path = str(sorted(ckpts, key=lambda x: x.stat().st_mtime)[-1])
    print(f"Using checkpoint: {checkpoint_path}")
    ck = safe_load_checkpoint(checkpoint_path, DEVICE)
    if 'config' in ck:
        for key, value in ck['config'].items():
            if hasattr(config, key):
                setattr(config, key, value)
    model = FacialGestureRegressionModel(config).to(DEVICE)
    if 'model_state_dict' in ck:
        model.load_state_dict(ck['model_state_dict'], strict=False)
    elif 'model' in ck:
        model.load_state_dict(ck['model'], strict=False)
    else:
        raise KeyError("Checkpoint doesn't contain model weights under 'model_state_dict' or 'model' keys")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    scaler_mean_path = os.path.join(checkpoint_dir, 'scaler_mean.npy')
    scaler_scale_path = os.path.join(checkpoint_dir, 'scaler_scale.npy')
    inference_scaler = StandardScaler()
    if os.path.exists(scaler_mean_path) and os.path.exists(scaler_scale_path):
        print(f"Loading scaler parameters from {checkpoint_dir}")
        try:
            inference_scaler.mean_ = np.load(scaler_mean_path)
            inference_scaler.scale_ = np.load(scaler_scale_path)
            print(f"Loaded scaler with {len(inference_scaler.mean_)} features")
        except Exception as e:
            print(f"Warning: Failed to load scaler parameters: {e}")
            print("Will proceed with basic normalization")
            inference_scaler = None
    else:
        print(f"Warning: Scaler files not found at {checkpoint_dir}")
        print("Will proceed with basic normalization")
        inference_scaler = None
    turker_df = None
    if args.turker_scores_path:
        turker_df = load_turker_scores(args.turker_scores_path)
    else:
        video_dir = os.path.dirname(args.video_path)
        dataset_root = os.path.dirname(video_dir)
        default_path = os.path.join(dataset_root, 'Labels', 'turker_scores_full_interview.csv')

        if os.path.exists(default_path):
            print(f"Found default turker scores at {default_path}")
            turker_df = load_turker_scores(default_path)
        else:
            print("No turker scores path provided and couldn't find default location")
            print("Ground truth comparison will be skipped")
    prediction, attention = process_video_for_inference(
        args.video_path, model, inference_scaler, config, DEVICE
    )

    if prediction is not None:
        print(f"Predicted {config.target_label} score: {prediction:.4f}")
        prediction_status = "PASS" if prediction >= 5.0 else "FAIL"
        print(f"Prediction status: {prediction_status}")
        if turker_df is not None:
            participant_id, participant_id_lower = extract_participant_id(args.video_path)
            ground_truth = get_ground_truth(turker_df, participant_id, participant_id_lower, config.target_label)
            if ground_truth is not None:
                print(f"Ground truth {config.target_label} score: {ground_truth:.4f}")
                ground_truth_status = "PASS" if ground_truth >= 5.0 else "FAIL"
                print(f"Ground truth status: {ground_truth_status}")
                print(f"Difference (predicted - ground truth): {prediction - ground_truth:.4f}")
                abs_error = abs(prediction - ground_truth)
                rel_error = abs_error / (abs(ground_truth) + 1e-8) * 100
                print(f"Absolute error: {abs_error:.4f}")
                print(f"Relative error: {rel_error:.2f}%")
                status_match = prediction_status == ground_truth_status
                print(f"Status match: {'YES' if status_match else 'NO'}")

            else:
                print(f"No ground truth found for participant {participant_id} and label {config.target_label}")
        if args.visualize and attention is not None:
            vis_path = os.path.join(os.path.dirname(args.video_path),
                                    f"{os.path.splitext(os.path.basename(args.video_path))[0]}_attention.png")
            visualize_attention(attention, save_path=vis_path)
            print("Attention highlights important facial landmarks for prediction")
    else:
        print("Failed to make prediction. Check if faces were detected in the video.")


if __name__ == '__main__':
    main()
