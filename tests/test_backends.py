from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest

from vidscope.backends.asr import AsrBackendFailure, FasterWhisperBackend
from vidscope.backends.media import FFmpegBackend, MediaBackendFailure
from vidscope.backends.ocr import OcrBackendFailure, TesseractBackend
from vidscope.backends.vad import SileroVadBackend


def _code(exc: BaseException) -> str:
    value: Any = getattr(exc, "error", exc)
    return str(getattr(value, "code", getattr(exc, "code", "")))


def test_ffmpeg_backend_uses_bounded_probe_audio_and_frame_commands(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    work = tmp_path / "work"
    calls: list[list[str]] = []

    def runner(command: list[str], **_: Any) -> Any:
        calls.append(command)
        output = Path(command[-1])
        if "ffprobe" in command[0]:
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"format": {"duration": "2"}, "streams": []}),
                stderr="",
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"nonempty")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    backend = FFmpegBackend(runner=runner, ffprobe_bin="ffprobe", ffmpeg_bin="ffmpeg")
    assert backend.probe(source)["format"]["duration"] == "2"
    audio = backend.extract_audio(source, output_directory=work)
    frames = backend.extract_frames(
        source,
        timestamps_seconds=(0.5, 1.5),
        max_frame_width=640,
        output_directory=work,
    )

    assert audio.stat().st_size > 0
    assert len(frames) == 2
    assert any("-show_streams" in command for command in calls)
    audio_command = next(command for command in calls if "-ar" in command)
    assert audio_command[
        audio_command.index("-vn") : audio_command.index("-vn") + 9
    ] == ["-vn", "-map", "0:a:0", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"]
    assert any(
        "scale" in token and "640" in token for command in calls for token in command
    )


def test_ffmpeg_backend_maps_failed_or_empty_output_to_typed_errors(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")

    def failed(_: list[str], **__: Any) -> Any:
        return types.SimpleNamespace(returncode=1, stdout="", stderr="decode failed")

    with pytest.raises(MediaBackendFailure) as failed_error:
        FFmpegBackend(runner=failed).probe(source)
    assert _code(failed_error.value) == "MEDIA_DECODE_FAILED"

    def empty(command: list[str], **_: Any) -> Any:
        if "ffprobe" in command[0]:
            return types.SimpleNamespace(returncode=0, stdout="{}", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(MediaBackendFailure) as empty_error:
        FFmpegBackend(runner=empty).extract_audio(source, output_directory=tmp_path)
    assert _code(empty_error.value) == "MEDIA_DECODE_FAILED"


def test_faster_whisper_backend_uses_benchmarked_local_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    calls: dict[str, Any] = {}

    class Segment:
        id = 1
        start = 0.0
        end = 1.0
        text = "hello"
        words: ClassVar[list[Any]] = [
            types.SimpleNamespace(start=0.0, end=0.5, word="hello", probability=0.9)
        ]

    class Model:
        def transcribe(self, path: str, **kwargs: Any) -> Any:
            calls["transcribe"] = (path, kwargs)
            return iter([Segment()]), types.SimpleNamespace(
                language="en", language_probability=0.99
            )

    def factory(*args: Any, **kwargs: Any) -> Model:
        calls["model"] = (args, kwargs)
        return Model()

    result = FasterWhisperBackend(model_factory=factory).transcribe(
        audio, language="en"
    )

    assert result.segments
    assert calls["model"] == (
        ("tiny.en",),
        {"device": "cpu", "compute_type": "int8", "cpu_threads": 4, "num_workers": 1},
    )
    assert calls["transcribe"][1] == {
        "language": "en",
        "beam_size": 5,
        "word_timestamps": True,
        "vad_filter": True,
    }


def test_faster_whisper_backend_maps_empty_and_load_failures(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")

    with pytest.raises(AsrBackendFailure) as load_error:
        FasterWhisperBackend(
            model_factory=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("missing")
            )
        ).transcribe(audio)
    assert _code(load_error.value) == "ASR_MODEL_UNAVAILABLE"

    class EmptyModel:
        def transcribe(self, *_args: Any, **_kwargs: Any) -> Any:
            return iter(()), types.SimpleNamespace(language="en")

    with pytest.raises(AsrBackendFailure) as empty_error:
        FasterWhisperBackend(
            model_factory=lambda *_args, **_kwargs: EmptyModel()
        ).transcribe(audio)
    assert _code(empty_error.value) == "INTERNAL_STAGE_FAILED"


def test_silero_vad_backend_clamps_and_sorts_intervals(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    module = types.SimpleNamespace(
        load_silero_vad=lambda onnx=False: object(),
        read_audio=lambda path, sampling_rate=16000: (path, sampling_rate),
        get_speech_timestamps=lambda audio_value, model, sampling_rate=16000, return_seconds=True: [
            {"start": 3.0, "end": 20.0},
            {"start": -2.0, "end": 1.0},
        ],
    )
    result = SileroVadBackend(module=module).detect(
        audio, start_seconds=0.0, end_seconds=10.0
    )

    assert result.intervals == [
        {"start_seconds": 0.0, "end_seconds": 1.0},
        {"start_seconds": 3.0, "end_seconds": 10.0},
    ]


def test_tesseract_backend_parses_tsv_and_retains_low_confidence(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")
    tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n5\t1\t1\t1\t1\t1\t2\t3\t4\t5\t12.5\tuncertain\n"
    seen: dict[str, Any] = {}

    def runner(command: list[str], **kwargs: Any) -> Any:
        seen["command"] = command
        seen["env"] = kwargs.get("env")
        return types.SimpleNamespace(returncode=0, stdout=tsv, stderr="")

    result = TesseractBackend(binary="tesseract", runner=runner).recognize(
        frame, language="eng"
    )

    assert result.rows[0]["text"] == "uncertain"
    assert result.rows[0]["confidence"] == 12.5
    assert seen["command"][-5:] == ["--oem", "1", "-l", "eng", "tsv"]


def test_tesseract_backend_maps_missing_malformed_and_empty_output(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")
    with pytest.raises(OcrBackendFailure) as missing:
        TesseractBackend(binary=None, runner=None).recognize(frame)
    assert _code(missing.value) == "OCR_UNAVAILABLE"

    def malformed(_: list[str], **__: Any) -> Any:
        return types.SimpleNamespace(returncode=0, stdout="not a tsv", stderr="")

    with pytest.raises(OcrBackendFailure) as malformed_error:
        TesseractBackend(binary="tesseract", runner=malformed).recognize(frame)
    assert _code(malformed_error.value) in {"OCR_UNAVAILABLE", "INTERNAL_STAGE_FAILED"}
