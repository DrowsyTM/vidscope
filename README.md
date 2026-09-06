# video-analyzer

Local-only, bounded video analysis through one Python API, CLI, and FastMCP tool.

## Verification

```bash
cd /home/dima/projects/video-analyzer
uv run pytest
uv run ruff check .
uv run mypy src tests
```

The opt-in live smoke uses the benchmark URL and portable Tesseract binaries:

```bash
cd /home/dima/projects/video-analyzer
VIDEO_ANALYZER_TESSERACT_BIN=/home/dima/video-tool-quality/tools/tesseract-local/usr/bin/tesseract \
TESSDATA_PREFIX=/home/dima/video-tool-quality/tools/tesseract-local/usr/share/tesseract-ocr/5/tessdata \
LD_LIBRARY_PATH=/home/dima/video-tool-quality/tools/tesseract-local/usr/lib/x86_64-linux-gnu \
YTDLP_IGNORE_CONFIG=1 \
uv run video-analyzer analyze-video \
  --source https://www.youtube.com/watch?v=7xTGNNLPyMI \
  --out /tmp/video-analyzer-smoke \
  --start-seconds 0 --end-seconds 180 \
  --task metadata --task transcript --task frames --task ocr
```

Inspect the emitted manifest URI and the run's `plan.json` and `manifest.json`. The
result should contain only bounded summaries and hashed artifact references; large
transcripts, media, frames, and OCR records stay in the run directory. The plan and
manifest must retain `cloud_policy: "deny"` and no cloud provider is selected.
