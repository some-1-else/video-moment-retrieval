"""Run Moment-DETR raw-video inference on a small Charades-STA JSONL subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import mean_iou, recall_at_iou


def add_moment_detr_to_path(repo_dir: Path) -> None:
    repo_dir = repo_dir.resolve()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


class OpenCVMomentDETRPredictor:
    """Wrapper around official Moment-DETR raw-video model code."""

    def __init__(
        self,
        repo_dir: str | Path,
        ckpt_path: str | Path,
        clip_model_name_or_path: str = "ViT-B/32",
        device: str = "cpu",
    ) -> None:
        repo_dir = Path(repo_dir)
        add_moment_detr_to_path(repo_dir)

        from moment_detr.span_utils import span_cxw_to_xx
        from run_on_video import clip
        from run_on_video.model_utils import build_inference_model
        from utils.tensor_utils import pad_sequences_1d

        self.clip_len = 2
        self.device = device
        self.span_cxw_to_xx = span_cxw_to_xx
        self.pad_sequences_1d = pad_sequences_1d
        self.tokenize = clip.tokenize
        self.normalizer = ImageNormalizer(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )

        self.clip_extractor, _ = clip.load(
            clip_model_name_or_path,
            device=device,
            jit=False,
        )
        self.model = self._build_model_with_legacy_torch_load(
            build_inference_model,
            ckpt_path,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def localize_moment(self, video_path: str | Path, query_list: list[str]) -> list[dict]:
        video_feats = self.encode_video(video_path)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(video_feats)
        if n_frames == 0:
            raise ValueError(f"No sampled frames extracted from {video_path}.")

        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)
        video_feats = torch.cat([video_feats, tef], dim=1)
        if n_frames > 75:
            raise ValueError(
                "The pretrained Moment-DETR positional embedding supports up to "
                f"75 two-second clips, but {n_frames} were extracted."
            )

        n_query = len(query_list)
        video_feats = video_feats.unsqueeze(0).repeat(n_query, 1, 1)
        video_mask = torch.ones(n_query, n_frames).to(self.device)
        query_feats = self.encode_text(query_list)
        query_feats, query_mask = self.pad_sequences_1d(
            query_feats,
            dtype=torch.float32,
            device=self.device,
            fixed_length=None,
        )
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)

        outputs = self.model(
            src_vid=video_feats,
            src_vid_mask=video_mask,
            src_txt=query_feats,
            src_txt_mask=query_mask,
        )
        prob = F.softmax(outputs["pred_logits"], -1)
        scores = prob[..., 0]
        pred_spans = outputs["pred_spans"]

        video_duration = n_frames * self.clip_len
        predictions = []
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            spans = self.span_cxw_to_xx(spans) * video_duration
            ranked = torch.cat([spans, score[:, None]], dim=1).tolist()
            ranked = sorted(ranked, key=lambda row: row[2], reverse=True)
            ranked = [[float(f"{value:.4f}") for value in row] for row in ranked]
            predictions.append(
                {
                    "query": query_list[idx],
                    "vid": str(video_path),
                    "pred_relevant_windows": ranked,
                    "sampled_clips": n_frames,
                    "moment_detr_video_duration": video_duration,
                }
            )

        return predictions

    @torch.no_grad()
    def encode_video(self, video_path: str | Path, batch_size: int = 60) -> torch.Tensor:
        frames = extract_two_second_clip_frames(video_path)
        frames = self.normalizer(frames / 255.0)
        features = []
        for start in range(0, len(frames), batch_size):
            batch = frames[start:start + batch_size].to(self.device)
            features.append(self.clip_extractor.encode_image(batch))
        return torch.cat(features, dim=0)

    @torch.no_grad()
    def encode_text(self, text_list: list[str], batch_size: int = 60) -> list[torch.Tensor]:
        features = []
        for start in range(0, len(text_list), batch_size):
            batch_text = text_list[start:start + batch_size]
            encoded_texts = self.tokenize(batch_text, context_length=77).to(self.device)
            output = self.clip_extractor.encode_text(encoded_texts)
            valid_lengths = (encoded_texts != 0).sum(1).tolist()
            last_hidden_states = output["last_hidden_state"]
            for idx, valid_length in enumerate(valid_lengths):
                features.append(last_hidden_states[idx, :valid_length])
        return features

    @staticmethod
    def _build_model_with_legacy_torch_load(build_inference_model, ckpt_path):
        original_torch_load = torch.load

        def legacy_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = legacy_torch_load
        try:
            return build_inference_model(str(ckpt_path))
        finally:
            torch.load = original_torch_load


def extract_two_second_clip_frames(video_path: str | Path) -> torch.Tensor:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"Invalid video metadata for {video_path}.")

        duration = frame_count / fps
        timestamps = np.arange(0.0, duration, 2.0)
        frames = []
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp * 1000.0))
            ok, frame = capture.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = resize_and_center_crop(frame, size=224)
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames could be read from {video_path}.")

    array = np.stack(frames).astype("float32")
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
    return tensor


class ImageNormalizer:
    def __init__(self, mean: tuple[float, float, float], std: tuple[float, float, float]):
        self.mean = torch.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = torch.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / (self.std + 1e-8)


def resize_and_center_crop(frame: np.ndarray, size: int = 224) -> np.ndarray:
    cv2 = _import_cv2()
    height, width = frame.shape[:2]
    if height >= width:
        new_width = size
        new_height = int(height * size / width)
    else:
        new_height = size
        new_width = int(width * size / height)

    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    y = max((new_height - size) // 2, 0)
    x = max((new_width - size) // 2, 0)
    return resized[y:y + size, x:x + size]


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if limit is not None and len(rows) >= limit:
                break
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}.") from error
    return rows


def run_moment_detr_subset(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.input_jsonl, limit=args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictor = OpenCVMomentDETRPredictor(
        repo_dir=args.repo_dir,
        ckpt_path=args.ckpt_path,
        clip_model_name_or_path=args.clip_model,
        device=args.device,
    )

    predictions_path = args.output_dir / "predictions.jsonl"
    prediction_windows: list[list[float]] = []
    ground_truth_windows: list[list[list[float]]] = []
    failed_queries = 0
    start_time = time.perf_counter()

    with predictions_path.open("w", encoding="utf-8") as output_file:
        for row_index, row in enumerate(rows):
            record = build_base_prediction_record(row, row_index, args.videos_dir)
            try:
                validate_input_row(row, row_index)
                video_path = Path(args.videos_dir) / f"{row['vid']}.mp4"
                if not video_path.exists():
                    raise FileNotFoundError(f"Missing video file: {video_path}")

                model_predictions = predictor.localize_moment(
                    video_path=video_path,
                    query_list=[str(row["query"])],
                )
                ranked_windows = model_predictions[0]["pred_relevant_windows"]
                if not ranked_windows:
                    raise ValueError("Moment-DETR returned no predicted windows.")

                top_window = [float(ranked_windows[0][0]), float(ranked_windows[0][1])]
                gt_windows = [
                    [float(window[0]), float(window[1])]
                    for window in row["relevant_windows"]
                ]

                record.update(
                    {
                        "status": "ok",
                        "predicted_window": top_window,
                        "predicted_score": float(ranked_windows[0][2]),
                        "ranked_predicted_windows": ranked_windows,
                        "error": "",
                    }
                )
                prediction_windows.append(top_window)
                ground_truth_windows.append(gt_windows)
            except Exception as error:
                failed_queries += 1
                record.update(
                    {
                        "status": "failed",
                        "predicted_window": None,
                        "predicted_score": None,
                        "ranked_predicted_windows": [],
                        "error": str(error),
                    }
                )

            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    inference_time_sec = time.perf_counter() - start_time
    evaluated_queries = len(prediction_windows)
    metrics = build_metrics(
        prediction_windows=prediction_windows,
        ground_truth_windows=ground_truth_windows,
        evaluated_queries=evaluated_queries,
        failed_queries=failed_queries,
        inference_time_sec=inference_time_sec,
    )
    run_config = {
        "input_jsonl": str(args.input_jsonl),
        "videos_dir": str(args.videos_dir),
        "repo_dir": str(args.repo_dir),
        "ckpt_path": str(args.ckpt_path),
        "clip_model": args.clip_model,
        "device": args.device,
        "limit": args.limit,
        "output_dir": str(args.output_dir),
    }

    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "run_config.json", run_config)

    return {
        "metrics": metrics,
        "run_config": run_config,
        "predictions_path": str(predictions_path),
    }


def build_base_prediction_record(
    row: dict[str, Any],
    row_index: int,
    videos_dir: str | Path,
) -> dict[str, Any]:
    video_id = str(row.get("vid", ""))
    return {
        "row_index": row_index,
        "qid": row.get("qid"),
        "vid": video_id,
        "query": row.get("query"),
        "video_path": str(Path(videos_dir) / f"{video_id}.mp4") if video_id else "",
        "duration": row.get("duration"),
        "relevant_windows": row.get("relevant_windows"),
    }


def validate_input_row(row: dict[str, Any], row_index: int) -> None:
    for field in ("qid", "vid", "query", "duration", "relevant_windows"):
        if field not in row:
            raise ValueError(f"Input row {row_index} is missing '{field}'.")
    if not isinstance(row["relevant_windows"], list) or not row["relevant_windows"]:
        raise ValueError(f"Input row {row_index} must contain relevant_windows.")
    first_window = row["relevant_windows"][0]
    if not isinstance(first_window, list) or len(first_window) != 2:
        raise ValueError(f"Input row {row_index} has invalid first relevant window.")


def build_metrics(
    prediction_windows: list[list[float]],
    ground_truth_windows: list[list[list[float]]],
    evaluated_queries: int,
    failed_queries: int,
    inference_time_sec: float,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "evaluated_queries": evaluated_queries,
        "failed_queries": failed_queries,
        "inference_time_sec": inference_time_sec,
    }

    if evaluated_queries == 0:
        metrics.update(
            {
                "R@1_IoU_0.3": 0.0,
                "R@1_IoU_0.5": 0.0,
                "R@1_IoU_0.7": 0.0,
                "mIoU": 0.0,
            }
        )
        return metrics

    metrics.update(
        {
            "R@1_IoU_0.3": recall_at_iou(
                prediction_windows,
                ground_truth_windows,
                threshold=0.3,
            ),
            "R@1_IoU_0.5": recall_at_iou(
                prediction_windows,
                ground_truth_windows,
                threshold=0.5,
            ),
            "R@1_IoU_0.7": recall_at_iou(
                prediction_windows,
                ground_truth_windows,
                threshold=0.7,
            ),
            "mIoU": mean_iou(prediction_windows, ground_truth_windows),
        }
    )
    return metrics


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Moment-DETR raw-video inference on Charades-STA subset JSONL."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("data/processed/charades_sta_moment_detr_test_subset.jsonl"),
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("data/raw/charades/videos"),
    )
    parser.add_argument("--repo-dir", type=Path, default=Path("external/moment_detr"))
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=Path("external/moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt"),
    )
    parser.add_argument("--clip-model", default="ViT-B/32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/moment_detr_charades_50"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_moment_detr_subset(args)
    metrics = result["metrics"]

    print(f"Saved predictions to: {result['predictions_path']}")
    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return 0


def _import_cv2():
    try:
        import cv2
    except ImportError as error:
        raise ImportError("opencv-python is required for this probe.") from error
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
