from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from vidscope.backends.source import (
    CaptionResolver,
    CaptionTrack,
    MediaAcquirer,
    SourceBackendFailure,
    SourceInspector,
    _source_path,
)


def _code(exc: BaseException) -> str:
    value: Any = getattr(exc, "error", exc)
    return str(getattr(value, "code", getattr(exc, "code", "")))


def test_windows_drive_path_is_parsed_as_local_source() -> None:
    source = r"C:\Users\runner\fixture.mp4"

    assert _source_path(source) == Path(source)


def test_windows_drive_authority_file_uri_is_parsed_as_local_source() -> None:
    source = "file://C:/Users/runner/fixture.mp4"

    assert _source_path(source) == Path("C:/Users/runner/fixture.mp4")


def test_non_ascii_file_authority_is_rejected() -> None:
    with pytest.raises(SourceBackendFailure) as exc_info:
        _source_path("file://é:/Users/runner/fixture.mp4")

    assert _code(exc_info.value) == "SOURCE_NOT_ALLOWED"


@pytest.mark.skipif(os.name != "nt", reason="Windows path semantics require Windows")
def test_windows_file_uri_is_parsed_as_local_source(tmp_path: Path) -> None:
    source_file = tmp_path / "fixture.mp4"

    assert _source_path(source_file.as_uri()) == source_file


def test_local_inspection_is_bounded_and_preserves_caption_stream_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip with spaces.mp4"
    source.write_bytes(b"fixture")

    def runner(command: list[str], **_: Any) -> Any:
        assert command[0] == "ffprobe"
        return {
            "streams": [
                {"codec_type": "video", "width": 640},
                {"codec_type": "subtitle", "tags": {"language": "en"}},
            ],
            "format": {"duration": "2.5"},
        }

    inspection = SourceInspector(runner=runner, ffprobe_bin="ffprobe").inspect(
        {"source": str(source)}
    )

    assert inspection.duration_seconds == 2.5
    assert inspection.caption_tracks[0].language == "en"
    assert inspection.metadata["source_kind"] == "local"
    assert inspection.metadata["streams"]


def test_url_inspection_disables_provider_config_and_playlists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    class Provider:
        def extract_info(self, source: str, **options: Any) -> dict[str, Any]:
            seen["source"] = source
            seen["options"] = options
            seen["env"] = os.environ.get("YTDLP_IGNORE_CONFIG")
            return {
                "id": "fixture",
                "duration": 3,
                "formats": [
                    {
                        "format_id": "18",
                        "vcodec": "h264",
                        "acodec": "aac",
                        "height": 360,
                    }
                ],
                "subtitles": {
                    "en": [{"url": "https://example.test/captions.vtt", "ext": "vtt"}]
                },
            }

    monkeypatch.delenv("YTDLP_IGNORE_CONFIG", raising=False)
    inspection = SourceInspector(yt_dlp_provider=Provider()).inspect(
        {"source": "https://example.test/watch?v=fixture"}
    )

    assert inspection.is_url is True
    assert inspection.duration_seconds == 3.0
    assert inspection.caption_tracks[0].kind == "manual"
    assert seen["env"] == "1"
    provider_options = seen["options"]["options"]
    assert provider_options["skip_download"] is True
    assert provider_options["noplaylist"] is True
    assert provider_options["postprocessors"] == []

    class PlaylistProvider:
        def extract_info(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {"_type": "playlist", "entries": [{"id": "one"}]}

    with pytest.raises(SourceBackendFailure) as error:
        SourceInspector(yt_dlp_provider=PlaylistProvider()).inspect(
            {"source": "https://example.test/playlist"}
        )
    assert _code(error.value) == "SOURCE_NOT_ALLOWED"


def test_caption_resolver_prefers_manual_exact_and_parses_provider_payload() -> None:
    tracks = [
        CaptionTrack("automatic", "en", "fake", "https://example.test/auto.vtt"),
        CaptionTrack("manual", "en-US", "fake", "https://example.test/locale.vtt"),
        CaptionTrack("manual", "en", "fake", "https://example.test/exact.vtt"),
    ]

    resolver = CaptionResolver(
        yt_dlp_provider=type(
            "Provider",
            (),
            {
                "fetch_caption": lambda _self, url: (
                    "WEBVTT\n\n00:00.000 --> 00:01.000\nhello"
                )
            },
        )()
    )
    inspection = type(
        "Inspection",
        (),
        {"source": "/tmp/fixture.mp4", "is_url": False, "caption_tracks": tracks},
    )()
    selected = resolver.resolve(inspection, {"language": "en"})

    assert selected is not None
    assert selected.kind == "manual"
    assert selected.language == "en"
    assert selected.segments[0]["text"] == "hello"


def test_url_acquisition_applies_format_stream_and_download_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class Downloader:
        def download(
            self, source: str, *, options: dict[str, Any], progress_hook: Any = None
        ) -> None:
            captured["source"] = source
            captured["options"] = options
            target = Path(str(options["outtmpl"]).replace("%(ext)s", "mp4"))
            target.write_bytes(b"bounded-media")
            if progress_hook:
                progress_hook(
                    {"downloaded_bytes": target.stat().st_size, "filename": str(target)}
                )

    inspection = type(
        "Inspection",
        (),
        {
            "source": "https://example.test/watch?v=fixture",
            "is_url": True,
            "formats": [
                {
                    "format_id": "video-only",
                    "vcodec": "h264",
                    "acodec": "none",
                    "height": 720,
                },
                {
                    "format_id": "muxed",
                    "vcodec": "h264",
                    "acodec": "aac",
                    "height": 360,
                    "filesize": 100,
                },
            ],
        },
    )()
    request = {
        "time_range": {"start_seconds": 1, "end_seconds": 3},
        "tasks": {"frames"},
        "max_download_bytes": 1024,
        "timeout_seconds": 30,
        "output_directory": tmp_path,
        "request_id": "run",
    }
    monkeypatch.delenv("YTDLP_IGNORE_CONFIG", raising=False)
    result = MediaAcquirer(downloader=Downloader()).acquire_window(inspection, request)

    assert Path(result).read_bytes() == b"bounded-media"
    assert captured["options"]["format"] == "muxed"
    assert captured["options"]["download_sections"] == ["*1.000000-3.000000"]
    assert captured["options"]["max_filesize"] == 1024
    assert os.environ.get("YTDLP_IGNORE_CONFIG") is None


@pytest.mark.parametrize(
    "forbidden_url",
    [
        "https://127.0.0.1/video.mp4",
        "https://localhost/video.mp4",
        "https://169.254.169.254/latest/meta-data",
        "https://10.0.0.1/video.mp4",
        "https://192.168.1.1/video.mp4",
        "http://example.com/video.mp4",
    ],
)
def test_ssrf_rejection_for_private_and_loopback_ips(forbidden_url: str) -> None:
    inspector = SourceInspector()
    with pytest.raises(SourceBackendFailure) as exc_info:
        inspector.inspect({"source": forbidden_url})
    assert _code(exc_info.value) in {"SOURCE_NOT_ALLOWED", "URL_SCHEME_NOT_ALLOWED"}


def test_settings_cookies_file_and_yt_dlp_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vidscope.settings import COOKIES_FILE_ENV, load_settings

    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    settings = load_settings({COOKIES_FILE_ENV: str(cookies)})
    assert settings.cookies_file == cookies

    captured: dict[str, Any] = {}

    class InspectorProvider:
        def extract_info(
            self,
            source: str,
            *,
            download: bool = False,
            options: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            captured["options"] = options or {}
            return {
                "id": "abc",
                "title": "sample",
                "duration": 10.0,
                "formats": [],
                "subtitles": {},
            }

    inspector = SourceInspector(
        yt_dlp_provider=InspectorProvider(),
        settings=settings,
    )
    inspector.inspect({"source": "https://example.com/watch?v=sample"})
    assert captured["options"].get("cookiefile") == str(cookies)
    if shutil.which("node"):
        assert "js_runtimes" in captured["options"]


def test_format_choice_pairs_separate_video_and_audio_dash_streams() -> None:
    from vidscope.backends.source import _format_choice

    formats = [
        {
            "format_id": "v-1080",
            "vcodec": "h264",
            "acodec": "none",
            "height": 1080,
            "filesize": 500,
        },
        {
            "format_id": "v-720",
            "vcodec": "h264",
            "acodec": "none",
            "height": 720,
            "filesize": 300,
        },
        {
            "format_id": "a-128",
            "vcodec": "none",
            "acodec": "aac",
            "abr": 128,
            "filesize": 100,
        },
        {
            "format_id": "a-64",
            "vcodec": "none",
            "acodec": "aac",
            "abr": 64,
            "filesize": 50,
        },
    ]

    # Both needed, no muxed format exists -> pairs best video + best audio
    chosen = _format_choice(formats, max_bytes=1000, need_video=True, need_audio=True)
    assert chosen == "v-1080+a-128"

    # Max bytes constraint forces lower tier
    chosen_constrained = _format_choice(
        formats, max_bytes=380, need_video=True, need_audio=True
    )
    assert chosen_constrained == "v-720+a-64"


def test_format_choice_prefers_requested_language_and_original_audio() -> None:
    from vidscope.backends.source import _format_choice

    formats = [
        {
            "format_id": "v-1080",
            "vcodec": "h264",
            "acodec": "none",
            "height": 1080,
            "filesize": 500,
        },
        {
            "format_id": "251-es-dub",
            "vcodec": "none",
            "acodec": "opus",
            "language": "es",
            "format_note": "Spanish, medium",
            "language_preference": -1,
            "abr": 155,
            "filesize": 150,
        },
        {
            "format_id": "251-en-orig",
            "vcodec": "none",
            "acodec": "opus",
            "language": "en-US",
            "format_note": "English (US) original (default), medium",
            "language_preference": 10,
            "abr": 136,
            "filesize": 130,
        },
    ]

    # Default/English request prefers original English audio even if Spanish dub has higher bitrate
    chosen = _format_choice(
        formats, max_bytes=1000, need_video=True, need_audio=True, language="en"
    )
    assert chosen == "v-1080+251-en-orig"

    # Audio only also selects English original
    chosen_audio = _format_choice(
        formats, max_bytes=1000, need_video=False, need_audio=True, language="en"
    )
    assert chosen_audio == "251-en-orig"

    # Explicit Spanish request prefers Spanish audio
    chosen_es = _format_choice(
        formats, max_bytes=1000, need_video=True, need_audio=True, language="es"
    )
    assert chosen_es == "v-1080+251-es-dub"


def test_source_is_youtube_hostname_validation() -> None:
    from vidscope.backends.source import _is_youtube_source

    # Valid YouTube URLs
    assert _is_youtube_source("https://www.youtube.com/watch?v=123") is True
    assert _is_youtube_source("https://youtube.com/watch?v=123") is True
    assert _is_youtube_source("https://m.youtube.com/watch?v=123") is True
    assert _is_youtube_source("https://youtu.be/123") is True
    assert _is_youtube_source("https://www.youtube-nocookie.com/embed/123") is True
    assert _is_youtube_source("youtube.com/watch?v=123") is True
    assert _is_youtube_source("//youtube.com/watch?v=123") is True

    # Non-HTTPS schemes
    assert _is_youtube_source("file://youtube.com/clip") is False
    assert _is_youtube_source("ftp://youtube.com/clip") is False
    assert _is_youtube_source("http://youtube.com/watch?v=123") is False

    # Non-YouTube URLs or files containing substrings
    assert _is_youtube_source("https://evil.com/youtube.com") is False
    assert _is_youtube_source("https://evil.com?v=youtube.com") is False
    assert _is_youtube_source("https://youtube.com.evil.com") is False
    assert _is_youtube_source("https://fakeyoutube.com") is False
    assert _is_youtube_source("file:///videos/youtube.com.mp4") is False
    assert _is_youtube_source("/tmp/youtube.com.mp4") is False
