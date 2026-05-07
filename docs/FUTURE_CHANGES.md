# EchoTrace: Future Changes and Improvements

Tracked roadmap for upcoming work on the EchoTrace pipeline.

---

## Threshold Normalization Across the Full Pipeline

Unify the decision threshold so that preprocessing, model output, and training all operate on the same calibrated scale. Currently the 0.5 sigmoid cutoff is used at inference but is not validated against the operating point seen during training.

> [!IMPORTANT]
> **Mentor guidance (May 2026):** A deepfake audio detector should ideally **not have a hard decision threshold at all.** The model should output a continuous spoof probability score and let the downstream consumer (user, API caller, or forensic analyst) decide their own operating point based on their tolerance for false positives vs false negatives. The current `> 0.5` threshold is a temporary convenience for the demo UI and evaluation scripts. Future versions should:
> - Return only the raw probability (0.0–1.0) as the primary output
> - Let the user set their own threshold via a slider in the Streamlit UI
> - Report EER-optimal threshold in evaluation reports as a *recommendation*, not a hard cutoff
> - Remove all hardcoded `"SPOOF" if prob > X` logic from the inference pipeline

- [ ] Audit every location where a threshold is applied (preprocessing clipping, sigmoid cutoff in `evaluate()`, Streamlit UI confidence display) and document the current values
- [ ] Compute the optimal threshold on the validation set using the EER operating point from the ROC curve after training
- [ ] Store the calibrated threshold in the saved checkpoint alongside model weights so it travels with the model
- [ ] Update `core/preprocess.py` scalar feature clipping ranges to match the distribution observed in training data rather than hardcoded [0, 1] assumptions
- [ ] Update `streamlit_app.py` to load the threshold from the checkpoint and use it as the default decision boundary instead of a fixed 0.5
- [ ] Update `scripts/evaluate_pc.py` and `scripts/evaluate_server.py` to use the stored threshold for consistent evaluation
- [ ] Add a config field or CLI flag to override the threshold at inference time for sensitivity tuning
- [ ] Write a unit test that asserts the threshold value round-trips correctly through save/load
- [ ] **Long-term:** Remove all hard thresholds from the inference pipeline entirely; return only continuous scores

---

## Audio Trim Mode in the UI

Add an interactive region selector to the Streamlit interface so users can isolate a specific segment of the uploaded audio for analysis, rather than always processing the first 4 seconds.

- [ ] Add a waveform visualization component to `streamlit_app.py` that renders the full uploaded audio
- [ ] Implement a dual-handle slider (start time, end time) beneath the waveform so users can select an arbitrary sub-region
- [ ] Wire the selected region to the `load_audio()` call so only the trimmed segment is passed through feature extraction
- [ ] Handle edge cases: selection shorter than the minimum required duration (pad with reflect), selection longer than 4 seconds (take the selected window as-is or warn)
- [ ] Display the selected duration and timestamp range in the UI
- [ ] Allow playback of the trimmed region before running detection so the user can verify their selection
- [ ] Ensure the trim boundaries are included in the analysis report/export so results are reproducible
- [ ] Test with very short selections (under 1 second) and very long files (over 10 minutes) to confirm stability

---

## 8 kHz Audio Support

Extend the pipeline to handle 8 kHz telephony and low-bandwidth audio natively, since a large portion of real-world forensic audio (call recordings, VoIP captures) is sampled at 8 kHz.

- [ ] Update `load_audio()` in `core/preprocess.py` to accept a configurable `target_sr` parameter and propagate it through the pipeline instead of hardcoding 16000
- [ ] Adjust mel spectrogram parameters (`n_mels`, `n_fft`, `hop_length`) in `build_feature_image()` to produce valid spectrograms at 8 kHz (Nyquist is 4 kHz, so current `n_fft=2048` at 8 kHz needs review)
- [ ] Recalculate LPC formant extraction ranges in `extract_scalar_features()` since F3 may exceed Nyquist at 8 kHz; clamp or adapt the formant search accordingly
- [ ] Update the HNR and CPP pitch lag ranges (`min_lag`, `max_lag`) to reflect the lower sample rate
- [ ] Add 8 kHz training data to the dataset pipeline (telephony corpora, downsampled versions of existing data, or both)
- [ ] Train a separate model head or fine-tune the existing model with mixed 8 kHz / 16 kHz batches and evaluate cross-rate generalization
- [ ] Add sample rate detection in the Streamlit UI and display a warning or auto-resample indicator when 8 kHz input is detected
- [ ] Benchmark detection accuracy on 8 kHz-only test sets and document any performance gap versus 16 kHz
- [ ] Ensure MUSAN noise augmentation resamples noise files to match the target sample rate of each batch

---

## Fake Advertisement / Investment Scam Detection

Extend EchoTrace beyond standalone audio clips to detect AI-generated voice in video advertisements, particularly fake investment promotions and fraudulent endorsement videos that use cloned celebrity voices.

- [ ] Build a video ingestion pipeline: extract audio track from MP4/MKV/WebM uploads using FFmpeg, then run existing EchoTrace analysis on the extracted audio
- [ ] Gather a curated dataset of fake AI-generated investment advertisement videos (crypto scams, stock pump schemes, fraudulent product endorsements)
- [ ] Gather corresponding real advertisement videos for the same categories to serve as negative examples
- [ ] Add a "Video Mode" tab in the Streamlit UI that accepts video file uploads alongside the existing audio-only mode
- [ ] Display synchronized results: show the video playback alongside the EchoTrace confidence timeline so users can see exactly which segments contain synthetic speech
- [ ] Implement segment-level analysis: split long advertisement audio into overlapping windows (e.g., 4s with 2s overlap) and report per-segment scores instead of a single aggregate score
- [ ] Add metadata extraction from the video (codec, bitrate, resolution, upload source) to provide additional forensic context
- [ ] Create a demo showcase page that runs EchoTrace on curated fake advertisement samples with side-by-side real vs fake comparisons
- [ ] Document detection accuracy on the advertisement-specific test set separately from the general deepfake benchmark

---

## Celebrity Voice Cloning Detection

Test and validate EchoTrace specifically against celebrity voice cloning attacks, which are the most common vector for social engineering and misinformation campaigns.

- [ ] Gather a test set of known celebrity voice clones (public examples from ElevenLabs, Resemble.AI, Bark, and open-source TTS models)
- [ ] Gather matching real celebrity audio samples from verified sources (interviews, podcasts, press conferences) for each target celebrity
- [ ] Run EchoTrace evaluation on the celebrity test set and report per-celebrity accuracy, EER, and failure cases
- [ ] Identify any celebrities or voice types where the model performs poorly and investigate whether specific TTS architectures are harder to detect
- [ ] Add a "Celebrity Voice Check" preset in the demo that loads pre-analyzed examples showing real vs cloned comparisons
- [ ] Test against multi-language celebrity clones (non-English voices) to evaluate cross-language robustness
- [ ] Document results in a dedicated evaluation report with confusion matrices per celebrity and per TTS engine

---

*Last updated: 2026-05-06*
