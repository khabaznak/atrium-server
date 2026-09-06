from __future__ import annotations

import base64
from pathlib import Path

import pytest

from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.media_tools_settings import SQLiteMediaToolsSettingsStore
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.tools.registry.native.media import build_ocr_extract_tool_definition, build_stt_transcribe_tool_definition, build_tts_render_tool_definition, build_analyze_image_tool_definition, analyze_image, analyze_task_image, verify_stt_recording
from alphonse.agent_v2.assets import AttachmentDescriptor, SQLiteAssetStore
from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue


def _runtime(tmp_path: Path):
    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Admin", users_root=tmp_path / "users")
    store = SQLiteMediaToolsSettingsStore(":memory:")
    return build_runtime_host(user_store=users, media_tools_settings_store=store), admin, store


def test_media_settings_save_invalidates_readiness_and_persists(tmp_path: Path) -> None:
    store = SQLiteMediaToolsSettingsStore(tmp_path / "media.sqlite3")
    saved = store.update("tts", {"enabled": True, "model_id": "local/qwen", "dtype": "float16", "speaker": "Alex"})
    assert saved.tts.enabled is True
    ready = store.mark_verification("tts", ready=True, preview="/tmp/sample.wav")
    assert ready.tts.available is True
    changed = store.update("tts", {"model_id": "local/new-qwen"})
    assert changed.tts.available is False
    assert SQLiteMediaToolsSettingsStore(tmp_path / "media.sqlite3").get().tts.model_id == "local/new-qwen"


def test_media_settings_unchanged_save_preserves_verification() -> None:
    store = SQLiteMediaToolsSettingsStore(":memory:")
    store.update("stt", {"enabled": True})
    ready = store.mark_verification("stt", ready=True, preview="hola")

    saved = store.update("stt", {"enabled": True, "executable_path": ready.stt.executable_path, "model": "base", "default_language": ""})

    assert saved.stt.available is True
    assert saved.stt.verification.preview == "hola"


def test_media_settings_validate_backend_configuration() -> None:
    store = SQLiteMediaToolsSettingsStore(":memory:")
    with pytest.raises(ValueError, match="media_tools_tts_dtype_invalid"):
        store.update("tts", {"enabled": True, "dtype": "int8"})
    with pytest.raises(ValueError, match="media_tools_ocr_base_url_invalid"):
        store.update("ocr", {"enabled": True, "ollama_base_url": "not-a-url"})


def test_daemon_media_tools_require_admin_and_verification_updates_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, admin, _ = _runtime(tmp_path)
    daemon = V2Daemon(runtime)
    with pytest.raises(PermissionError, match="admin_required"):
        daemon.media_tools_settings(actor_user_id="not-admin")
    daemon.save_media_tools_settings(actor_user_id=admin.user_id, kind="tts", values={"enabled": True})
    monkeypatch.setattr("alphonse.agent_v2.daemon.verify_tts", lambda settings, sample_text: {"output": {"file_path": "/tmp/voice.wav"}, "exception": None})
    result = daemon.verify_media_tools(actor_user_id=admin.user_id, kind="tts")
    assert result["settings"]["tts"]["available"] is True
    assert result["result"]["exception"] is None


def test_recorded_stt_verification_marks_ready_and_keeps_transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, admin, _ = _runtime(tmp_path)
    daemon = V2Daemon(runtime)
    daemon.save_media_tools_settings(actor_user_id=admin.user_id, kind="stt", values={"enabled": True})
    captured: dict[str, object] = {}

    def fake_verify(settings, *, audio_base64, mime_type, duration_ms):  # noqa: ANN001
        captured.update({"settings": settings, "audio_base64": audio_base64, "mime_type": mime_type, "duration_ms": duration_ms})
        return {"output": {"text": "hola Alphonse", "segments": []}, "exception": None}

    monkeypatch.setattr("alphonse.agent_v2.daemon.verify_stt_recording", fake_verify)
    result = daemon.verify_stt_recording(actor_user_id=admin.user_id, audio_base64=base64.b64encode(b"audio").decode(), mime_type="audio/webm", duration_ms=1_000)

    assert captured["mime_type"] == "audio/webm"
    assert result["settings"]["stt"]["available"] is True
    assert result["settings"]["stt"]["verification"]["preview"] == "hola Alphonse"


def test_recorded_stt_verification_failure_does_not_mark_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, admin, _ = _runtime(tmp_path)
    daemon = V2Daemon(runtime)
    daemon.save_media_tools_settings(actor_user_id=admin.user_id, kind="stt", values={"enabled": True})
    monkeypatch.setattr("alphonse.agent_v2.daemon.verify_stt_recording", lambda *args, **kwargs: {"output": None, "exception": {"code": "whisper_cli_not_found", "message": "Whisper CLI was not found."}})

    result = daemon.verify_stt_recording(actor_user_id=admin.user_id, audio_base64=base64.b64encode(b"audio").decode(), mime_type="audio/webm", duration_ms=1_000)

    assert result["settings"]["stt"]["available"] is False
    assert result["settings"]["stt"]["verification"]["error"] == "Whisper CLI was not found."


def test_recorded_stt_verification_validates_audio_and_removes_temporary_file(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = SQLiteMediaToolsSettingsStore(":memory:").get().stt
    captured: dict[str, Path] = {}

    def fake_transcribe(_settings, *, asset_path: str):  # noqa: ANN001
        source = Path(asset_path)
        captured["source"] = source
        assert source.is_file()
        assert source.read_bytes() == b"recorded audio"
        return {"output": {"text": "hola", "segments": []}, "exception": None}

    monkeypatch.setattr("alphonse.agent_v2.core.tools.registry.native.media.transcribe_stt", fake_transcribe)
    result = verify_stt_recording(settings, audio_base64=base64.b64encode(b"recorded audio").decode(), mime_type="audio/webm;codecs=opus", duration_ms=1_500)

    assert result["output"]["text"] == "hola"
    assert not captured["source"].exists()
    assert verify_stt_recording(settings, audio_base64="", mime_type="audio/webm", duration_ms=1_000)["exception"]["code"] == "stt_recording_empty"
    assert verify_stt_recording(settings, audio_base64=base64.b64encode(b"x").decode(), mime_type="video/webm", duration_ms=1_000)["exception"]["code"] == "stt_recording_type_unsupported"
    assert verify_stt_recording(settings, audio_base64=base64.b64encode(b"x").decode(), mime_type="audio/webm", duration_ms=30_001)["exception"]["code"] == "stt_recording_duration_invalid"


def test_media_tools_are_not_capd_registered() -> None:
    tools = build_native_tool_registry()
    assert tools.get("tts_render") is None
    assert tools.get("stt_transcribe") is None
    assert tools.get("ocr_extract_text") is None
    assert tools.get("analyze_image") is None


def test_media_tool_contracts_exist_but_are_disabled_until_attachment_phase() -> None:
    store = SQLiteMediaToolsSettingsStore(":memory:")
    settings = store.get()
    assert build_tts_render_tool_definition(settings.tts).enabled is False
    assert build_stt_transcribe_tool_definition(settings.stt).enabled is False
    assert build_ocr_extract_tool_definition(settings.ocr).enabled is False


def test_verified_vision_tool_analyzes_only_task_owned_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SQLiteMediaToolsSettingsStore(":memory:")
    store.update("ocr", {"enabled": True})
    settings = store.mark_verification("ocr", ready=True).ocr
    asset_store = SQLiteAssetStore(tmp_path / "assets.sqlite3", tmp_path / "assets")
    asset = asset_store.register_bytes(owner_user_id="alex", descriptor=AttachmentDescriptor("photo.jpg", "image/jpeg", 3), content=b"jpg", source="telegram")
    task = TaskState(user="alex", metadata={"asset_ids": [asset.asset_id]})
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue())
    monkeypatch.setattr("alphonse.agent_v2.core.tools.registry.native.media.analyze_image", lambda settings, asset_path, question, empty_code="image_analysis_empty": {"output": {"text": f"seen: {question}"}, "exception": None})

    result = analyze_task_image({"asset_id": asset.asset_id, "question": "What is shown?"}, context=context, settings=settings, asset_store=asset_store)

    assert result["output"]["text"] == "seen: Free OCR."
    with pytest.raises(ValueError, match="task_attachment_not_found"):
        analyze_task_image({"asset_id": "other", "question": "What is shown?"}, context=context, settings=settings, asset_store=asset_store)
    assert build_analyze_image_tool_definition(settings, asset_store).enabled is True


def test_attachment_manifest_is_task_visible_without_local_path() -> None:
    task = TaskState(
        goal="What is in this?",
        metadata={"attachments": [{"asset_id": "asset-1", "filename": "photo.jpg", "mime_type": "image/jpeg", "kind": "photo", "ingestion_status": "registered", "caption": "What is in this?", "path": "/private/secret/photo.jpg"}]},
    )

    prompt = task.to_markdown_prompt()

    assert "Asset ID: asset-1" in prompt
    assert "photo.jpg" in prompt
    assert "/private/secret/photo.jpg" not in prompt


def test_daemon_refreshes_vision_tool_after_verification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, admin, _ = _runtime(tmp_path)
    daemon = V2Daemon(runtime)
    daemon.save_media_tools_settings(actor_user_id=admin.user_id, kind="ocr", values={"enabled": True})
    assert runtime.core.tools.get("analyze_image") is None
    monkeypatch.setattr("alphonse.agent_v2.daemon.verify_ocr", lambda settings, sample_path: {"output": {"text": "ok"}, "exception": None})

    daemon.verify_media_tools(actor_user_id=admin.user_id, kind="ocr", sample="image.jpg")

    assert runtime.core.tools.get("analyze_image") is not None


def test_qwen_image_analysis_uses_caption_question_and_returns_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status_code = 200
        def json(self): return {"message": {"content": "A red bicycle."}}
    captured = {}
    monkeypatch.setattr("alphonse.agent_v2.core.tools.registry.native.media.requests.post", lambda url, json, timeout: captured.update({"url": url, "json": json, "timeout": timeout}) or Response())
    settings = SQLiteMediaToolsSettingsStore(":memory:").get().ocr
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"jpg")

    result = analyze_image(settings, asset_path=str(image), question="What is in this image?")

    assert result["output"]["text"] == "A red bicycle."
    assert captured["json"]["messages"][0]["content"] == "What is in this image?"
    assert captured["json"]["messages"][0]["images"]
    assert captured["json"]["think"] is False
    assert isinstance(result["output"]["latency_ms"], int)


def test_dedicated_deepseek_ocr_uses_its_minimal_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status_code = 200
        def json(self): return {"message": {"content": "Extracted text"}}

    captured = {}
    monkeypatch.setattr("alphonse.agent_v2.core.tools.registry.native.media.requests.post", lambda url, json, timeout: captured.update({"json": json}) or Response())
    settings = SQLiteMediaToolsSettingsStore(":memory:").get().ocr
    image = tmp_path / "document.png"
    image.write_bytes(b"png")

    result = __import__("alphonse.agent_v2.core.tools.registry.native.media", fromlist=["extract_ocr"]).extract_ocr(settings, asset_path=str(image))

    assert result["output"]["text"] == "Extracted text"
    assert captured["json"]["messages"][0]["content"] == "Free OCR."


def test_desktop_receipt_uses_ollama_with_manifest_id_and_preserves_prices(tmp_path, monkeypatch):
    runtime, admin, settings_store = _runtime(tmp_path)
    settings_store.update("ocr", {"enabled": True})
    settings = settings_store.mark_verification("ocr", ready=True).ocr
    project = runtime.project_store.create_project(name="Receipts", root_path=str(tmp_path / "project"), owner_user_id=admin.user_id)
    source = tmp_path / "receipt.jpg"
    source.write_bytes(b"receipt image")
    attachments = V2Daemon(runtime).copy_desktop_project_files(user=admin.user_id, project_id=project.project_id, source_paths=[str(source)])
    assert attachments[0]["asset_id"]
    task = TaskState(user=admin.user_id, project_id=project.project_id, metadata={"attachments": attachments})
    captured = {}

    class Response:
        status_code = 200
        def json(self):
            return {"message": {"content": "1 ORDEN PASTOR 95.00\n1 DR PEPPER 35.00"}}

    def post(url, json, timeout):
        captured.update(json)
        return Response()

    monkeypatch.setattr("alphonse.agent_v2.core.tools.registry.native.media.requests.post", post)
    result = analyze_task_image({"asset_id": attachments[0]["asset_id"], "question": "¿Cuánto por pastor y Dr Pepper?"}, context=ToolExecutionContext(task=task, messages=InMemoryMessageQueue()), settings=settings, asset_store=runtime.asset_store)
    assert result["exception"] is None
    assert "95.00" in result["output"]["text"]
    assert "35.00" in result["output"]["text"]
    assert captured["messages"][0]["content"] == "Free OCR."
    assert base64.b64decode(captured["messages"][0]["images"][0]) == b"receipt image"

    foreign_task = TaskState(user="someone-else", metadata={"attachments": attachments})
    with pytest.raises(ValueError, match="task_image_not_found_or_forbidden"):
        analyze_task_image({"asset_id": attachments[0]["asset_id"], "question": "Read"}, context=ToolExecutionContext(task=foreign_task, messages=InMemoryMessageQueue()), settings=settings, asset_store=runtime.asset_store)


def test_ocr_timeout_is_reported_separately_from_disabled_backend(tmp_path, monkeypatch):
    import requests
    image = tmp_path / "receipt.jpg"
    image.write_bytes(b"jpg")
    def timeout(*args, **kwargs):
        raise requests.Timeout()
    monkeypatch.setattr("alphonse.agent_v2.core.tools.registry.native.media.requests.post", timeout)
    result = analyze_image(SQLiteMediaToolsSettingsStore(":memory:").get().ocr, asset_path=str(image), question="Read")
    assert result["exception"]["code"] == "image_analysis_timeout"
