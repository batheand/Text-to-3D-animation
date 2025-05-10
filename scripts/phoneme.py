

#!/usr/bin/env python3
"""
Callable class for extracting phoneme segments from audio using Allophant.

Example usage:
    python phoneme.py --wav input.wav --output data/phonemes
"""
import argparse
import os
import torch
import torchaudio
import numpy as np
import csv
import json
from allophant.estimator import Estimator
from allophant.dataset_processing import Batch
import soundfile as sf

class PhonemeTimelineExtractor:
    def __init__(
        self,
        model_id: str = "kgnlp/allophant",
        lang: str = "en",
        device: str = "cpu",
        max_segment_duration: float = None,
        drop_silence: bool = False,
        silence_token: str = '-'
    ):
        # Load model and phoneme inventory
        self.device = device
        self.model, self.attribute_indexer = Estimator.restore(model_id, device=device)
        self.inventory = self.attribute_indexer.phoneme_inventory(lang)
        self.sample_rate = self.model.sample_rate

        # Fixed language ID = 0 for inference
        self.lang_id = 0

        # Compute frame shift from conv strides (fallback to 20 ms)
        try:
            conv_layers = (
                self.model.model._acoustic_model._model.feature_extractor.conv_layers
            )
            strides = [layer.conv.stride[0] for layer in conv_layers]
            downsample = int(np.prod(strides))
            self.frame_shift = downsample / self.sample_rate
        except Exception:
            self.frame_shift = 0.02

        # Segment filtering options
        self.max_segment_duration = max_segment_duration
        self.drop_silence = drop_silence
        self.silence_token = silence_token

    def __call__(
        self,
        wav_path: str = None,
        waveform: torch.Tensor = None,
        orig_sr: int = None,
        threshold: float = None
    ) -> list:
        """
        Extract phoneme segments from audio. Provide either wav_path or waveform+orig_sr.
        threshold: average confidence cutoff for segments
        Returns list of dicts: phoneme, start_time, end_time, confidence.
        """
        # Load audio if path provided
        if wav_path is not None:
            try:
                waveform, orig_sr = torchaudio.load(wav_path)
            except RuntimeError:
                # Fallback to soundfile if torchaudio has no backend
                data, sr = sf.read(wav_path)
                waveform = torch.from_numpy(data).unsqueeze(0)
                orig_sr = sr
        if waveform is None or orig_sr is None:
            raise ValueError("Provide wav_path or both waveform and orig_sr")

        # Mono and resample if needed
        waveform = waveform[:1]
        if orig_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_sr, self.sample_rate
            )
        waveform = waveform.to(self.device)

        # Prepare batch: fixed lang_id = 0
        lengths = torch.tensor([waveform.shape[1]], dtype=torch.long)
        lang_ids = torch.zeros(1, dtype=torch.long)
        batch = Batch(waveform, lengths, lang_ids)

        # Model prediction
        outputs = self.model.predict(
            batch.to(self.device),
            self.attribute_indexer.composition_feature_matrix(self.inventory).to(self.device)
        )

        # Extract time-distributed logits (T, 1, C) -> (T, C)
        raw = outputs.outputs["phoneme"]       # (T_frames, 1, N_phonemes)
        logits = raw.permute(1, 0, 2)[0].cpu()    # (T_frames, N_phonemes)
        probs = torch.softmax(logits, dim=-1).numpy()
        ids = logits.argmax(dim=-1).tolist()
        frame_probs = [probs[i, pid] for i, pid in enumerate(ids)]

        # Collapse into segments
        segments = []
        start = 0
        prev_id = ids[0]
        for i, curr_id in enumerate(ids[1:], start=1):
            if curr_id != prev_id:
                seg = self._make_segment(prev_id, start, i, frame_probs, threshold)
                if seg:
                    segments.append(seg)
                prev_id = curr_id
                start = i
        # Last segment
        seg = self._make_segment(prev_id, start, len(ids), frame_probs, threshold)
        if seg:
            segments.append(seg)

        return segments

    def _make_segment(self, pid, start, end, frame_probs, threshold):
        confidence = float(np.mean(frame_probs[start:end]))
        duration = (end - start) * self.frame_shift
        phoneme = self.inventory[pid]
        # Apply filters
        if threshold is not None and confidence < threshold:
            return None
        if self.drop_silence and phoneme == self.silence_token:
            return None
        if self.max_segment_duration and duration > self.max_segment_duration:
            return None
        return {
            "phoneme": phoneme,
            "start_time": start * self.frame_shift,
            "end_time": end * self.frame_shift,
            "confidence": confidence
        }

    def to_csv(self, segments: list, csv_path: str):
        """Write segments to a CSV file."""
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["phoneme", "start_time", "end_time", "confidence"]
            )
            writer.writeheader()
            for seg in segments:
                writer.writerow(seg)

    def to_json(self, segments: list, json_path: str = None) -> str:
        """Serialize segments to JSON. Optionally write to file."""
        json_str = json.dumps(segments, indent=2)
        if json_path:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w") as f:
                f.write(json_str)
        return json_str

def main():
    parser = argparse.ArgumentParser(description="Extract phoneme timeline from WAV")
    parser.add_argument("--wav",      required=True, help="Input WAV file")
    parser.add_argument("--output",   required=True, help="Output CSV file path")
    parser.add_argument("--threshold", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--max_duration", type=float, default=0.3, help="Max segment duration (s)")
    parser.add_argument("--drop_silence", action="store_true", help="Drop silence segments")
    parser.add_argument("--silence_token", default='-', help="Token for silence")
    args = parser.parse_args()

    extractor = PhonemeTimelineExtractor(
        model_id="kgnlp/allophant",
        lang="en",
        device="cpu",
        max_segment_duration=args.max_duration,
        drop_silence=args.drop_silence,
        silence_token=args.silence_token
    )

    segments = extractor(wav_path=args.wav, threshold=args.threshold)
    extractor.to_csv(segments, args.output)
    print(f"Extracted {len(segments)} segments to {args.output}")

if __name__ == "__main__":
    main()