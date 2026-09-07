# Test Fixtures Provenance & Licenses

This directory contains deterministic test fixtures used in testing and benchmarking `video-analyzer`.

## Files

### `speech_sample.wav`
- **Format**: 16,000 Hz, 16-bit mono uncompressed PCM WAV.
- **Duration**: 3.0 seconds.
- **Content**: Deterministic synthetic 440 Hz tone burst with bounded active intervals (0.5s to 2.0s) and flanking silence for VAD and audio extraction pipeline verification.
- **License**: Creative Commons Zero v1.0 Universal (CC0 1.0 Public Domain Dedication).
