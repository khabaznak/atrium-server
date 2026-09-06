"""Foreground v2 daemon host."""

from __future__ import annotations

import signal
import json
import fcntl
import logging
import mimetypes
import os
import sqlite3
import shutil
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any
from uuid import uuid4

from alphonse.agent_v2.core.core import CoreUiEvent
from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.core.io import OutboundSelector
from alphonse.agent_v2.core.io import channel_address_from_metadata
from alphonse.agent_v2.core.io import SQLiteCommunicationThreadStore
from alphonse.agent_v2.core.io import upsert_provider_user_mapping
from alphonse.agent_v2.core.io import project_snapshot_to_outbox
from alphonse.agent_v2.core.io import SQLiteOutboundStore
from alphonse.agent_v2.core.messages import SQLiteMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.scheduled_tasks import schedule_summary
from alphonse.agent_v2.database import connect_database, transaction
from alphonse.agent_v2.integrations import SQLiteIntegrationStore
from alphonse.agent_v2.inference_settings import SQLiteInferenceSettingsStore
from alphonse.agent_v2.ipc import V2DaemonClient
from alphonse.agent_v2.ipc import V2DaemonServer
from alphonse.agent_v2.ipc import default_socket_path
from alphonse.agent_v2.runtime import V2RuntimeHost
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.runtime import start_runtime_integrations
from alphonse.agent_v2.runtime import stop_runtime_integrations
from alphonse.agent_v2.runtime import refresh_runtime_inference
from alphonse.agent_v2.runtime import refresh_runtime_identity_resolver
from alphonse.agent_v2.inference_settings import validate_and_save_inference_settings
from alphonse.agent_v2.services.project_sessions import ProjectSessionKey
from alphonse.agent_v2.services.project_sessions import SQLiteProjectSessionStore
from alphonse.agent_v2.agent_config import AgentConfigStore
from alphonse.agent_v2.interfaces.a2ui import ALPHONSE_DESKTOP_CATALOG_ID
from alphonse.agent_v2.interfaces.a2ui import A2UiAdapter
from alphonse.agent_v2.interfaces.a2ui import question_id_from_surface
from alphonse.agent_v2.interfaces.a2ui import surface_id_for_question
from alphonse.agent_v2.interfaces.ag_ui import AgUiAdapter
from alphonse.agent_v2.services.scheduled_worker import ScheduledTaskWorker
from alphonse.agent_v2.services.killswitch import KillSwitchAuditStore
from alphonse.agent_v2.services.killswitch import KillSwitchCoordinator
from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.assets import AttachmentDescriptor
from alphonse.agent_v2.web_tools_settings import WebToolsSettings
from alphonse.agent_v2.code_mode_settings import CodeModeSettings
from alphonse.agent_v2.code_mode_settings import SQLiteCodeModeSettingsStore
from alphonse.agent_v2.memory_settings import MemorySettings
from alphonse.agent_v2.memory_settings import SQLiteMemorySettingsStore
from alphonse.agent_v2.web_tools_settings import SQLiteWebToolsSettingsStore
from alphonse.agent_v2.media_tools_settings import SQLiteMediaToolsSettingsStore
from alphonse.agent_v2.core.tools.registry.native.media import verify_ocr, verify_stt, verify_stt_recording, verify_tts
from alphonse.agent_v2.runtime import refresh_runtime_web_tools
from alphonse.agent_v2.runtime import refresh_runtime_media_tools
from alphonse.agent_v2.runtime import refresh_runtime_artifacts
from alphonse.agent_v2.core.tools.registry.native.web import execute_web_fetch, execute_web_search
from alphonse.agent_v2.assets import SQLiteAssetStore
from alphonse.agent_v2.artifacts import SQLiteArtifactStore
from alphonse.agent_v2.conversations import SQLiteConversationStore, legacy_ledger_events
from alphonse.agent_v2.automations import EventAutomationStore
from alphonse.agent_v2.storage_migration import migrate_legacy_databases
from alphonse.agent_v2.retention import prune_operational_data


logger = logging.getLogger(__name__)


def _claim_single_instance_lock(daemon_id: str) -> Any:
    lock_path = default_socket_path().with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        try:
            V2DaemonClient(default_socket_path(), timeout_sec=0.5).ping()
        except Exception as exc:
            raise RuntimeError("alphonse_v2_daemon_lock_held") from exc
        raise RuntimeError("alphonse_v2_daemon_already_running")
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(daemon_id)
    lock_file.flush()
    return lock_file


@dataclass
class V2Daemon:
    runtime: V2RuntimeHost
    poll_interval_sec: float = 0.05
    inbound_max_attempts: int = 5
    event_store: EventAutomationStore | None = None
    daemon_id: str = ""

    def __post_init__(self) -> None:
        schedule_db_path = str(getattr(self.runtime.schedule_store, "db_path", ":memory:"))
        self.event_store = self.event_store or (
            EventAutomationStore(":memory:")
            if schedule_db_path == ":memory:"
            else EventAutomationStore(schedule_db_path)
        )
        self.daemon_id = str(self.daemon_id or "").strip() or f"daemon-{uuid4().hex[:12]}"
        if hasattr(self.runtime.queue, "lease_owner"):
            self.runtime.queue.lease_owner = self.daemon_id
        self._stop = threading.Event()
        self._processor_thread: threading.Thread | None = None
        self._lock_file: Any | None = None
        self._lifecycle_lock = threading.RLock()
        self._stopped = False
        self._last_processor_error = ""
        self._active_work_lock = threading.RLock()
        self._active_work: dict[str, str] = {}
        self.kill_switch = KillSwitchCoordinator()
        audit_path = str(getattr(self.runtime.outbox, "db_path", ":memory:"))
        self.kill_switch_audit = KillSwitchAuditStore(audit_path)
        self.runtime.core.cancellation_checker = self.kill_switch.is_cancelled
        self.runtime.core.active_task_callback = self._activate_kill_switch_task
        self.runtime.inbound_router.kill_switch_handler = self._handle_kill_switch_command
        self.runtime.inbound_router.active_task_lookup = self.active_work
        self._activity_status: dict[str, str] = {"state": "idle", "updated_at": datetime.now(timezone.utc).isoformat()}
        self._event_lock = threading.RLock()
        self._activity_event_sequence = 0
        self._legacy_activity_cursor = 0
        self._activity_event_journal: list[dict[str, Any]] = []
        self._ui_event_lock = threading.RLock()
        self._ui_event_sequence = 0
        self._ui_event_journal: list[dict[str, Any]] = []
        self._desktop_capabilities: dict[str, set[str]] = {}
        self._desktop_surfaces: dict[tuple[str, str, str], set[str]] = {}
        self._desktop_progress_surfaces: dict[tuple[str, str], set[str]] = {}
        self._desktop_progress_closures: dict[tuple[str, str], set[str]] = {}
        self._last_retention_at: datetime | None = None
        self._ag_ui = AgUiAdapter(question_store=self.runtime.question_store)
        self._a2ui = A2UiAdapter()
        self.scheduler = ScheduledTaskWorker(
            store=self.runtime.schedule_store,
            messages=self.runtime.queue,
            worker_id=self.daemon_id,
            on_message_queued=lambda: None,
            on_direct_delivery=self._deliver_scheduled_reminder,
        )
        self.ipc = V2DaemonServer(self)

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._stopped:
                raise RuntimeError("alphonse_v2_daemon_stopped")
            if self._processor_thread is not None and self._processor_thread.is_alive():
                return
            self._acquire_single_instance_lock()
        try:
            self._stop.clear()
            self._ensure_home_projects()
            self._migrate_blank_project_records()
            reclaim_expired = getattr(self.runtime.queue, "reclaim_expired", None)
            if callable(reclaim_expired):
                reclaim_expired()
            self._run_retention_if_due(force=True)
            # Publish the health socket before optional providers initialize so
            # clients can distinguish a live daemon from a failed startup.
            self.ipc.start()
            self._align_configured_provider_addresses()
            start_runtime_integrations(
                self.runtime,
                on_outbox_delivered=self._on_outbox_delivered,
                on_outbox_failed=self._on_outbox_failed,
            )
            self.scheduler.start()
            self._processor_thread = threading.Thread(target=self._process_loop, name="alphonse-v2-core", daemon=True)
            self._processor_thread.start()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        with self._lifecycle_lock:
            if self._stopped:
                return
            self._stopped = True
        self._stop.set()
        self.ipc.stop()
        self.scheduler.stop()
        stop_runtime_integrations(self.runtime)
        if self._processor_thread is not None and self._processor_thread.is_alive():
            self._processor_thread.join(timeout=5)
        self._release_single_instance_lock()

    def restart_integrations(self) -> None:
        self._align_configured_provider_addresses()
        start_runtime_integrations(
            self.runtime,
            on_outbox_delivered=self._on_outbox_delivered,
            on_outbox_failed=self._on_outbox_failed,
        )

    def _activate_kill_switch_task(self, queued: Any, task: Any) -> None:
        message = getattr(queued, "message", None)
        self.kill_switch.activate(
            message_id=str(getattr(queued, "message_id", "") or ""),
            task_id=str(getattr(task, "task_id", "") or ""),
            user_id=str(getattr(task, "user", "") or ""),
            project_id=str(getattr(task, "project_id", "") or ""),
            metadata=dict(getattr(message, "metadata", {}) or {}),
        )

    def _handle_kill_switch_command(self, *, user_id: str, address: Any, arguments: str) -> str:
        if str(arguments or "").strip():
            return "Usage: /killswitch"
        result = self.trigger_killswitch(actor_user_id=user_id, source=address.to_dict())
        if result["status"] == "denied":
            return "Access denied: administrator privileges are required."
        if result["status"] == "no_active_task":
            return "Kill switch checked: no active task is running."
        notice = "queued" if result.get("notification_outbox_id") else "could not be queued"
        return f"Kill switch engaged. Active task cancelled; owner security notice {notice}. Audit: {result['audit_id']}."

    def trigger_killswitch(self, *, actor_user_id: str, source: dict[str, Any] | None = None) -> dict[str, Any]:
        """Cancel the one task actively executing in this daemon, if authorized."""
        actor = str(actor_user_id or "").strip()
        source_value = dict(source or {})
        if not self.runtime.user_store.is_admin(actor):
            return {"authorized": False, "status": "denied", "active_task": {}, "audit_id": ""}
        active = self.kill_switch.request_cancel()
        if active is None:
            audit_id = self.kill_switch_audit.record(actor_user_id=actor, source=source_value, active=None, status="no_active_task")
            return {"authorized": True, "status": "no_active_task", "active_task": {}, "audit_id": audit_id}
        notification_outbox_id, notification_error = self._queue_killswitch_notice(active)
        audit_id = self.kill_switch_audit.record(
            actor_user_id=actor, source=source_value, active=active, status="cancel_requested",
            notification_outbox_id=notification_outbox_id, notification_error=notification_error,
        )
        return {
            "authorized": True, "status": "cancel_requested", "active_task": {
                "message_id": active.message_id, "task_id": active.task_id, "user_id": active.user_id,
                "project_id": active.project_id,
            }, "notification_outbox_id": notification_outbox_id, "notification_error": notification_error,
            "audit_id": audit_id,
        }

    def _queue_killswitch_notice(self, active: Any) -> tuple[str, str]:
        fallback = channel_address_from_metadata(active.metadata)
        resolved = self.runtime.identity_resolver.resolve_outbound_address(
            alphonse_user_id=active.user_id, fallback_address=fallback,
        )
        if not resolved.resolved or resolved.address is None:
            return "", str(resolved.reason or "owner_delivery_unavailable")
        try:
            outbound = self.runtime.outbox.enqueue(
                address=resolved.address,
                message="Your task was eliminated for security reasons. Please try later or contact Alphonse’s Admin.",
                kind="security_notice", audience_user_id=active.user_id, task_id=active.task_id,
                project_id=active.project_id, metadata={"killswitch": True, "target_message_id": active.message_id},
            )
        except Exception as exc:
            return "", f"{type(exc).__name__}: {exc}"
        return outbound.outbox_message_id, ""

    def _align_configured_provider_addresses(self) -> None:
        records = self.runtime.integration_store.list_enabled()
        for record in records:
            same_provider = [item for item in records if item.provider_key == record.provider_key]
            if len(same_provider) == 1:
                self.runtime.user_store.align_provider_addresses(
                    provider_key=record.provider_key,
                    integration_id=record.integration_id,
                )

    def update_inference_settings(self, *, provider_key: str, model_id: str) -> dict[str, str]:
        settings = validate_and_save_inference_settings(
            self.runtime.inference_settings_store,
            provider_key=provider_key,
            model_id=model_id,
        )
        refresh_runtime_inference(self.runtime, settings)
        return settings.to_dict()

    def onboarding_status(self) -> dict[str, object]:
        return self.runtime.user_store.status()

    def onboard(self, *, display_name: str, users_root: str, import_v1: bool = False) -> dict[str, object]:
        store = self.runtime.user_store
        # Kept as an ignored IPC parameter so an already-installed desktop client
        # can complete onboarding after the v1 code has been removed.
        del import_v1
        admin = store.onboard(display_name=display_name, users_root=users_root)
        imported = {}
        self.runtime.user = admin.user_id
        self._ensure_home_projects()
        self._migrate_blank_project_records()
        migrated = self._migrate_legacy_local(admin.user_id)
        # Legacy-local ownership becomes resolvable only after the ownership
        # migration, so run the idempotent provenance pass once more.
        self._ensure_home_projects()
        self._migrate_blank_project_records()
        self.runtime.asset_store.migrate_to_user_directories()
        refresh_runtime_identity_resolver(self.runtime)
        return {"admin_user": admin.to_dict(), "migration": {**imported, **migrated}, "users_root": str(store.users_root())}

    def _migrate_legacy_local(self, admin_user_id: str) -> dict[str, int]:
        """Idempotently claim records created before canonical-user onboarding."""
        projects = self.runtime.project_store.migrate_owner("local", admin_user_id)
        sessions = self.runtime.project_session_store.migrate_user("local", admin_user_id)
        integrations = 0
        schedules = 0
        for record in self.runtime.integration_store.list():
            config = dict(record.config)
            if str(config.get("owner_user_id") or "") != "local":
                continue
            config["owner_user_id"] = admin_user_id
            self.runtime.integration_store.upsert(integration_id=record.integration_id, provider_key=record.provider_key, display_name=record.display_name, enabled=record.enabled, config=config, secrets=dict(record.secrets))
            integrations += 1
        if getattr(self.runtime.schedule_store, "db_path", ":memory:") != ":memory:":
            with connect_database(self.runtime.schedule_store.db_path) as conn:
                schedules = conn.execute("UPDATE v2_scheduled_tasks SET owner_user_id=? WHERE owner_user_id='local'", (admin_user_id,)).rowcount
        return {"local_projects_migrated": projects, "local_sessions_migrated": sessions, "local_integrations_migrated": integrations, "local_schedules_migrated": schedules}

    def current_user(self) -> dict[str, object]:
        admin = self.runtime.user_store.admin_user()
        return {"user": admin.to_dict() if admin else None, "onboarded": admin is not None}

    def _admin_user_id(self, requested: str = "") -> str:
        admin = self.runtime.user_store.admin_user()
        if admin is None:
            if self.runtime.user_store.is_ephemeral and str(requested or "").strip():
                return str(requested).strip()
            raise RuntimeError("v2_onboarding_required")
        return admin.user_id

    def list_users(self) -> list[dict[str, object]]:
        self.runtime.user_store.normalize_duplicate_addresses({record.integration_id for record in self.runtime.integration_store.list()})
        return [{**user.to_dict(), "addresses": [address.to_dict() for address in self.runtime.user_store.list_addresses(user.user_id)], "aliases": self.runtime.user_store.list_aliases(user.user_id)} for user in self.runtime.user_store.list_users()]

    def create_user(self, *, display_name: str, role: str = "member") -> dict[str, object]:
        user = self.runtime.user_store.create_user(display_name=display_name, role=role)
        self._ensure_home_projects()
        self._migrate_blank_project_records()
        return user.to_dict()

    def scheduled_tasks(
        self,
        *,
        actor_user_id: str = "",
        owner_user_id: str = "",
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        owner = self._scheduled_task_owner(actor_user_id=actor_user_id, requested_owner_user_id=owner_user_id)
        tasks = self.runtime.schedule_store.list_tasks(owner_user_id=owner, status=status, limit=limit)
        return [{**task.to_dict(), "latest_execution": self._latest_scheduled_execution(task.scheduled_task_id)} for task in tasks]

    def scheduled_task_executions(self, *, actor_user_id: str = "", scheduled_task_id: str, limit: int = 100) -> list[dict[str, object]]:
        self._scheduled_task_for_actor(actor_user_id=actor_user_id, scheduled_task_id=scheduled_task_id)
        return [item.to_dict() for item in self.runtime.schedule_store.list_executions(scheduled_task_id=scheduled_task_id, limit=limit)]

    def update_scheduled_task(self, *, actor_user_id: str = "", scheduled_task_id: str, name: str, prompt: str) -> dict[str, object]:
        self._scheduled_task_for_actor(actor_user_id=actor_user_id, scheduled_task_id=scheduled_task_id)
        return self.runtime.schedule_store.update_task(scheduled_task_id, name=name, prompt=prompt).to_dict()

    def pause_scheduled_task(self, *, actor_user_id: str = "", scheduled_task_id: str) -> dict[str, object]:
        self._scheduled_task_for_actor(actor_user_id=actor_user_id, scheduled_task_id=scheduled_task_id)
        return self.runtime.schedule_store.pause_task(scheduled_task_id).to_dict()

    def resume_scheduled_task(self, *, actor_user_id: str = "", scheduled_task_id: str) -> dict[str, object]:
        self._scheduled_task_for_actor(actor_user_id=actor_user_id, scheduled_task_id=scheduled_task_id)
        return self.runtime.schedule_store.resume_task(scheduled_task_id).to_dict()

    def cancel_scheduled_task(self, *, actor_user_id: str = "", scheduled_task_id: str) -> dict[str, object]:
        task = self._scheduled_task_for_actor(actor_user_id=actor_user_id, scheduled_task_id=scheduled_task_id)
        if getattr(task, "status", "") not in {"active", "paused"}:
            raise ValueError("scheduled_task_not_cancellable")
        return self.runtime.schedule_store.cancel_task(scheduled_task_id).to_dict()

    def delete_scheduled_task(self, *, actor_user_id: str = "", scheduled_task_id: str) -> bool:
        self._scheduled_task_for_actor(actor_user_id=actor_user_id, scheduled_task_id=scheduled_task_id)
        return self.runtime.schedule_store.delete_task(scheduled_task_id)

    def _scheduled_task_owner(self, *, actor_user_id: str, requested_owner_user_id: str) -> str:
        actor = str(actor_user_id or self._admin_user_id()).strip() or self._admin_user_id()
        if self.runtime.user_store.get_user(actor) is None:
            raise ValueError("scheduled_task_actor_not_found")
        requested = str(requested_owner_user_id or actor).strip() or actor
        if requested != actor and not self.runtime.user_store.is_admin(actor):
            raise PermissionError("scheduled_task_owner_forbidden")
        if self.runtime.user_store.get_user(requested) is None:
            raise KeyError("scheduled_task_owner_not_found")
        return requested

    def _scheduled_task_for_actor(self, *, actor_user_id: str, scheduled_task_id: str) -> object:
        task = self.runtime.schedule_store.get_task(scheduled_task_id)
        if task is None:
            raise KeyError(f"scheduled_task_not_found: {scheduled_task_id}")
        owner = self._scheduled_task_owner(actor_user_id=actor_user_id, requested_owner_user_id=task.owner_user_id)
        if task.owner_user_id != owner:
            raise PermissionError("scheduled_task_owner_forbidden")
        return task

    def _latest_scheduled_execution(self, scheduled_task_id: str) -> dict[str, object] | None:
        executions = self.runtime.schedule_store.list_executions(scheduled_task_id=scheduled_task_id, limit=1)
        return executions[0].to_dict() if executions else None

    def update_user(self, user_id: str, **values: Any) -> dict[str, object]:
        return self.runtime.user_store.update_user(user_id, display_name=values.get("display_name"), role=values.get("role"), is_active=values.get("is_active")).to_dict()

    def delete_user(self, user_id: str, *, confirmation: str) -> dict[str, int]:
        self._admin_user_id()
        user = self.runtime.user_store.get_user(user_id)
        if user is None:
            raise KeyError("user_not_found")
        if str(confirmation or "") != user.user_id:
            raise ValueError("delete_confirmation_must_match_user_id")
        roots = self.runtime.project_store.delete_owned_by(user.user_id)
        self.runtime.project_session_store.delete_user(user.user_id)
        for root in roots:
            path = Path(root)
            if path.exists() and self.runtime.user_store.users_root() in path.parents:
                shutil.rmtree(path)
        schedules = self._delete_user_schedule_data(user.user_id)
        questions = self._delete_user_question_data(user.user_id)
        inbound = self._delete_user_inbound_data(user.user_id)
        outbound = self._delete_user_outbound_data(user.user_id)
        deleted = self.runtime.user_store.delete_user(user.user_id)
        return {"deleted": int(deleted), "projects": len(roots), "schedules": schedules, "questions": questions, "inbound": inbound, "outbound": outbound}

    def _delete_user_schedule_data(self, user_id: str) -> int:
        store = self.runtime.schedule_store
        with store._connect() as conn:
            ids = [str(row[0]) for row in conn.execute("SELECT scheduled_task_id FROM v2_scheduled_tasks WHERE owner_user_id=?", (user_id,)).fetchall()]
            if ids:
                conn.executemany("DELETE FROM v2_scheduled_task_executions WHERE scheduled_task_id=?", [(task_id,) for task_id in ids])
            return conn.execute("DELETE FROM v2_scheduled_tasks WHERE owner_user_id=?", (user_id,)).rowcount

    def _delete_user_question_data(self, user_id: str) -> int:
        store = self.runtime.question_store
        with store._connect() as conn:
            rows = conn.execute("SELECT question_id FROM v2_questions WHERE respondent_user_id=? OR originator_user_id=?", (user_id, user_id)).fetchall()
            ids = [str(row[0]) for row in rows]
            if ids:
                conn.executemany("DELETE FROM v2_task_dependencies WHERE question_id=?", [(question_id,) for question_id in ids])
            conn.execute("DELETE FROM v2_task_checkpoints WHERE owner_id=?", (user_id,))
            return conn.execute("DELETE FROM v2_questions WHERE respondent_user_id=? OR originator_user_id=?", (user_id, user_id)).rowcount

    def _delete_user_inbound_data(self, user_id: str) -> int:
        queue = self.runtime.queue
        connect = getattr(queue, "_connect", None)
        if not callable(connect):
            return 0
        with connect() as conn:
            return conn.execute("DELETE FROM v2_inbound_messages WHERE user_id=?", (user_id,)).rowcount

    def _delete_user_outbound_data(self, user_id: str) -> int:
        with self.runtime.outbox._connect() as conn:
            return conn.execute("DELETE FROM v2_outbox WHERE audience_user_id=?", (user_id,)).rowcount

    def user_context(self, user_id: str) -> dict[str, str]:
        return {"user_id": user_id, "content": self.runtime.user_store.read_user_context(user_id)}

    def save_user_context(self, user_id: str, content: str) -> dict[str, str]:
        return {"user_id": user_id, "path": self.runtime.user_store.write_user_context(user_id, content)}

    def bind_user_address(self, **values: Any) -> dict[str, object]:
        return self.runtime.user_store.bind_address(**values).to_dict()

    def remove_user_address(self, address_id: str) -> bool:
        return self.runtime.user_store.remove_address(address_id)

    def set_user_aliases(self, *, user_id: str, aliases: list[str]) -> dict[str, object]:
        return {"user_id": user_id, "aliases": self.runtime.user_store.set_aliases(user_id, aliases)}

    def pending_access_requests(self) -> list[dict[str, object]]:
        return [request.to_dict() for request in self.runtime.user_store.list_access_requests()]

    def approve_access_request(self, *, request_id: str, display_name: str = "", user_id: str = "") -> dict[str, object]:
        request, address = self.runtime.user_store.approve_access_request(
            request_id,
            display_name=display_name,
            user_id=user_id,
        )
        self._ensure_home_projects()
        self._migrate_blank_project_records()
        if request.provider_key == "telegram":
            record = self.runtime.integration_store.get(request.integration_id)
            if record is None:
                raise ValueError("access_request_integration_not_found")
            config = dict(record.config)
            allowed = {str(value).strip() for value in config.get("allowed_chat_ids", []) if str(value).strip()}
            allowed.add(request.channel_target)
            self.runtime.integration_store.upsert(
                integration_id=record.integration_id,
                provider_key=record.provider_key,
                display_name=record.display_name,
                enabled=record.enabled,
                config={**config, "allowed_chat_ids": sorted(allowed)},
                secrets=record.secrets,
            )
            refresh_runtime_identity_resolver(self.runtime)
            self.restart_integrations()
        return {"request": request.to_dict(), "address": address.to_dict()}

    def _ensure_home_projects(self) -> None:
        """Reconcile the protected default project for every active user."""
        for user in self.runtime.user_store.list_users():
            if not user.is_active:
                continue
            root = self.runtime.user_store.managed_project_root(user.user_id) / "home"
            project = self.runtime.project_store.ensure_home_project(user.user_id, root_path=str(root))
            migrate = getattr(self.runtime.core.memory, "migrate_legacy_project_ledgers", None)
            if callable(migrate):
                migrate(project.project_id, include_generic=True)
        migrate = getattr(self.runtime.core.memory, "migrate_legacy_project_ledgers", None)
        if callable(migrate):
            for project in self.runtime.project_store.list_manageable_projects("", requester_is_admin=True):
                migrate(project.project_id)

    def _migrate_blank_project_records(self) -> None:
        """Assign legacy unscoped records to their known owner's Home project."""
        path = str(getattr(self.runtime.queue, "db_path", ":memory:"))
        if path == ":memory:":
            return
        homes = {user.user_id: self.runtime.project_store.home_project(user.user_id) for user in self.runtime.user_store.list_users()}
        ids = {user: project.project_id for user, project in homes.items() if project is not None}
        if not ids:
            return
        with connect_database(path) as conn:
            tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            def update(table: str, owner_column: str) -> None:
                if table not in tables:
                    return
                for owner, project_id in ids.items():
                    conn.execute(
                        f"UPDATE {table} SET project_id=? WHERE (project_id='' OR project_id IS NULL) AND {owner_column}=?",
                        (project_id, owner),
                    )
            update("v2_inbound_messages", "user_id")
            update("v2_conversation_events", "owner_user_id")
            update("v2_scheduled_tasks", "owner_user_id")
            update("v2_automations", "owner_user_id")
            update("v2_task_checkpoints", "owner_id")
            update("v2_outbox", "audience_user_id")
            if "v2_project_sessions" in tables:
                for owner, project_id in ids.items():
                    conn.execute(
                        "UPDATE v2_project_sessions SET active_project_id=?, project_name='Home' WHERE active_project_id='' AND alphonse_user_id=?",
                        (project_id, owner),
                    )
            if "v2_scheduled_task_executions" in tables:
                conn.execute("UPDATE v2_scheduled_task_executions SET project_id=(SELECT project_id FROM v2_scheduled_tasks t WHERE t.scheduled_task_id=v2_scheduled_task_executions.scheduled_task_id) WHERE project_id='' OR project_id IS NULL")
            if "v2_automation_executions" in tables:
                # Executions derive project provenance from their automation; no independent project column exists.
                pass
            if "v2_questions" in tables:
                conn.execute("UPDATE v2_questions SET project_id=(SELECT project_id FROM v2_task_checkpoints c WHERE c.task_id=v2_questions.task_id) WHERE project_id='' OR project_id IS NULL")
            if "v2_inbound_messages" in tables:
                conn.execute(
                    "UPDATE v2_inbound_messages SET metadata_json=json_set(COALESCE(metadata_json, '{}'), '$.routing_disposition', 'pdca_task') "
                    "WHERE json_extract(COALESCE(metadata_json, '{}'), '$.routing_disposition') IS NULL"
                )
            unresolved: dict[str, int] = {}
            for table in (
                "v2_inbound_messages", "v2_conversation_events", "v2_scheduled_tasks",
                "v2_automations", "v2_task_checkpoints", "v2_outbox", "v2_questions",
            ):
                if table in tables:
                    count = conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE project_id='' OR project_id IS NULL"
                    ).fetchone()[0]
                    if count:
                        unresolved[table] = int(count)
            if unresolved:
                logger.warning("project provenance migration left rows unresolved: %s", unresolved)

    def reject_access_request(self, *, request_id: str) -> dict[str, object]:
        return self.runtime.user_store.reject_access_request(request_id).to_dict()

    def settings(self) -> dict[str, object]:
        return self.runtime.user_store.status()

    def save_settings(
        self,
        *,
        users_root: str,
        timezone_name: str = "",
        mirror_automation_messages_to_preferred_channel: bool | None = None,
    ) -> dict[str, object]:
        timezone_value = self.runtime.user_store.set_timezone(timezone_name) if str(timezone_name).strip() else self.runtime.user_store.timezone()
        if mirror_automation_messages_to_preferred_channel is not None:
            self.runtime.user_store.set_mirror_automation_messages_to_preferred_channel(
                mirror_automation_messages_to_preferred_channel
            )
        saved_users_root = self.runtime.user_store.set_users_root(users_root)
        self.runtime.asset_store.migrate_to_user_directories()
        return {
            "users_root": saved_users_root,
            "timezone": timezone_value,
            "mirror_automation_messages_to_preferred_channel": self.runtime.user_store.mirror_automation_messages_to_preferred_channel(),
            "warning_repository_path": "/Alphonse/" in str(users_root),
        }

    def timezone_settings(self, *, actor_user_id: str) -> dict[str, str]:
        self._require_admin(actor_user_id)
        return {"timezone": self.runtime.user_store.timezone()}

    def save_timezone_settings(self, *, actor_user_id: str, timezone_name: str) -> dict[str, str]:
        self._require_admin(actor_user_id)
        return {"timezone": self.runtime.user_store.set_timezone(timezone_name)}

    def memory_settings(self, *, actor_user_id: str) -> dict[str, object]:
        self._require_admin(actor_user_id)
        return self.runtime.memory_settings_store.get().to_dict()

    def save_memory_settings(self, *, actor_user_id: str, values: dict[str, Any]) -> dict[str, object]:
        self._require_admin(actor_user_id)
        current = self.runtime.memory_settings_store.get()
        saved = self.runtime.memory_settings_store.save(MemorySettings(
            max_ledger_bytes=values.get("max_ledger_bytes", current.max_ledger_bytes),
            compaction_summary_max_words=values.get("compaction_summary_max_words", current.compaction_summary_max_words),
        ))
        return saved.to_dict()

    def list_artifacts(self, *, actor_user_id: str) -> list[dict[str, Any]]:
        actor = str(actor_user_id or "").strip()
        if not actor: raise PermissionError("artifact_manager_required")
        owner = "" if self.runtime.user_store.is_admin(actor) else actor
        return [item.to_dict() for item in self.runtime.artifact_store.list(owner_user_id=owner)]

    def update_artifact(self, *, actor_user_id: str, artifact_id: str, name: str, description: str) -> dict[str, Any]:
        self._require_artifact_manager(actor_user_id, self.runtime.artifact_store.get(artifact_id))
        saved = self.runtime.artifact_store.update_metadata(artifact_id, name=name, description=description)
        refresh_runtime_artifacts(self.runtime)
        return saved.to_dict()

    def set_artifact_enabled(self, *, actor_user_id: str, artifact_id: str, enabled: bool) -> dict[str, Any]:
        self._require_artifact_manager(actor_user_id, self.runtime.artifact_store.get(artifact_id))
        saved = self.runtime.artifact_store.set_enabled(artifact_id, enabled)
        refresh_runtime_artifacts(self.runtime)
        return saved.to_dict()

    def delete_artifact(self, *, actor_user_id: str, artifact_id: str) -> dict[str, Any]:
        self._require_artifact_manager(actor_user_id, self.runtime.artifact_store.get(artifact_id))
        self.runtime.artifact_store.delete(artifact_id)
        refresh_runtime_artifacts(self.runtime)
        return {"deleted": artifact_id}

    def web_tools_settings(self, *, actor_user_id: str) -> dict[str, object]:
        self._require_admin(actor_user_id)
        return self.runtime.web_tools_settings_store.get().to_dict()

    def code_mode_settings(self, *, actor_user_id: str) -> dict[str, object]:
        self._require_admin(actor_user_id)
        return self.runtime.code_mode_settings_store.get().to_dict()

    def save_code_mode_settings(self, *, actor_user_id: str, values: dict[str, Any], acknowledge_unsafe: bool = False) -> dict[str, object]:
        self._require_admin(actor_user_id)
        current = self.runtime.code_mode_settings_store.get()
        saved = CodeModeSettings(
            enabled=bool(values.get("enabled", current.enabled)), docker_bin=str(values.get("docker_bin", current.docker_bin)), image=str(values.get("image", current.image)),
            timeout_seconds=values.get("timeout_seconds", current.timeout_seconds), max_tool_calls=values.get("max_tool_calls", current.max_tool_calls), max_parallel_calls=values.get("max_parallel_calls", current.max_parallel_calls),
            memory_mb=values.get("memory_mb", current.memory_mb), cpu_count=values.get("cpu_count", current.cpu_count), pid_limit=values.get("pid_limit", current.pid_limit), tmpfs_mb=values.get("tmpfs_mb", current.tmpfs_mb),
            network_disabled=bool(values.get("network_disabled", current.network_disabled)), read_only_filesystem=bool(values.get("read_only_filesystem", current.read_only_filesystem)),
            run_as_non_root=bool(values.get("run_as_non_root", current.run_as_non_root)), drop_all_capabilities=bool(values.get("drop_all_capabilities", current.drop_all_capabilities)), no_new_privileges=bool(values.get("no_new_privileges", current.no_new_privileges)),
        )
        if saved.weakened_protections and not acknowledge_unsafe:
            raise ValueError("code_mode_unsafe_configuration_confirmation_required")
        return self.runtime.code_mode_settings_store.save(saved).to_dict()

    def verify_code_mode(self, *, actor_user_id: str) -> dict[str, object]:
        self._require_admin(actor_user_id)
        result = self.runtime.core.program_runner.verify()
        settings = self.runtime.code_mode_settings_store.mark_verification(ready=bool(result.get("ready")), error=str(result.get("error") or ""))
        return {"result": result, "settings": settings.to_dict()}

    def save_web_tools_settings(self, *, actor_user_id: str, values: dict[str, Any]) -> dict[str, object]:
        self._require_admin(actor_user_id)
        current = self.runtime.web_tools_settings_store.get()
        saved = self.runtime.web_tools_settings_store.save(WebToolsSettings(
            enabled=bool(values.get("enabled", current.enabled)), searxng_base_url=str(values.get("searxng_base_url", current.searxng_base_url)),
            search_timeout_seconds=values.get("search_timeout_seconds", current.search_timeout_seconds), fetch_timeout_seconds=values.get("fetch_timeout_seconds", current.fetch_timeout_seconds),
            fetch_max_chars=values.get("fetch_max_chars", current.fetch_max_chars),
        ))
        refresh_runtime_web_tools(self.runtime)
        return saved.to_dict()

    def verify_web_tools(self, *, actor_user_id: str, kind: str) -> dict[str, Any]:
        self._require_admin(actor_user_id)
        settings = self.runtime.web_tools_settings_store.get()
        if kind == "search": return execute_web_search({"query": "Alphonse SearXNG verification", "limit": 1}, settings=settings)
        if kind == "fetch": return execute_web_fetch({"url": "https://example.com", "max_chars": 200}, settings=settings)
        raise ValueError("web_tools_verify_kind_invalid")

    def media_tools_settings(self, *, actor_user_id: str) -> dict[str, object]:
        self._require_admin(actor_user_id)
        return self.runtime.media_tools_settings_store.get().to_dict()

    def save_media_tools_settings(self, *, actor_user_id: str, kind: str, values: dict[str, Any]) -> dict[str, object]:
        self._require_admin(actor_user_id)
        saved = self.runtime.media_tools_settings_store.update(kind, values)
        refresh_runtime_media_tools(self.runtime)
        return saved.to_dict()

    def verify_media_tools(self, *, actor_user_id: str, kind: str, sample: str = "") -> dict[str, Any]:
        self._require_admin(actor_user_id)
        settings = self.runtime.media_tools_settings_store.get()
        if kind == "tts": result = verify_tts(settings.tts, sample_text=sample or "Alphonse text-to-speech verification.")
        elif kind == "stt": result = verify_stt(settings.stt, sample_path=sample)
        elif kind == "ocr": result = verify_ocr(settings.ocr, sample_path=sample)
        else: raise ValueError("media_tools_kind_invalid")
        exception = result.get("exception") if isinstance(result, dict) else {"message": "verification_failed"}
        output = result.get("output") if isinstance(result, dict) else {}
        preview = str((output or {}).get("text") or (output or {}).get("file_path") or "")
        if kind == "ocr" and isinstance(output, dict) and isinstance(output.get("latency_ms"), (int, float)):
            preview = f"Verified in {float(output['latency_ms']) / 1000:.1f}s: {preview}"
        error = str((exception or {}).get("message") or "") if isinstance(exception, dict) else ""
        details = (exception or {}).get("details") if isinstance(exception, dict) else {}
        detail_error = str((details or {}).get("error") or "") if isinstance(details, dict) else ""
        if detail_error:
            error = f"{error}: {detail_error}"
        saved = self.runtime.media_tools_settings_store.mark_verification(kind, ready=not bool(exception), error=error, preview=preview)
        refresh_runtime_media_tools(self.runtime)
        return {"result": result, "settings": saved.to_dict()}

    def verify_stt_recording(
        self,
        *,
        actor_user_id: str,
        audio_base64: str,
        mime_type: str,
        duration_ms: int,
    ) -> dict[str, Any]:
        self._require_admin(actor_user_id)
        settings = self.runtime.media_tools_settings_store.get()
        result = verify_stt_recording(
            settings.stt,
            audio_base64=audio_base64,
            mime_type=mime_type,
            duration_ms=duration_ms,
        )
        exception = result.get("exception") if isinstance(result, dict) else {"message": "verification_failed"}
        output = result.get("output") if isinstance(result, dict) else {}
        preview = str((output or {}).get("text") or "")
        error = str((exception or {}).get("message") or "") if isinstance(exception, dict) else ""
        details = (exception or {}).get("details") if isinstance(exception, dict) else {}
        detail_error = str((details or {}).get("error") or "") if isinstance(details, dict) else ""
        if detail_error:
            error = f"{error}: {detail_error}"
        saved = self.runtime.media_tools_settings_store.mark_verification(
            "stt", ready=not bool(exception), error=error, preview=preview,
        )
        refresh_runtime_media_tools(self.runtime)
        return {"result": result, "settings": saved.to_dict()}

    def _require_admin(self, actor_user_id: str) -> None:
        actor = str(actor_user_id or "").strip()
        if not actor or not self.runtime.user_store.is_admin(actor):
            raise PermissionError("admin_required")

    def _require_artifact_manager(self, actor_user_id: str, record: Any) -> None:
        if record is None:
            raise KeyError("artifact_not_found")
        actor = str(actor_user_id or "").strip()
        if actor != str(record.owner_user_id) and not self.runtime.user_store.is_admin(actor):
            raise PermissionError("artifact_manager_required")

    def list_agent_config(self) -> list[dict[str, str]]:
        return [document.to_dict(include_content=False) for document in self.runtime.agent_config_store.list_documents()]

    def read_agent_config(self, file_name: str) -> dict[str, str]:
        return self.runtime.agent_config_store.read(file_name).to_dict()

    def save_agent_config(self, file_name: str, content: str) -> dict[str, str]:
        return self.runtime.agent_config_store.save(file_name, content).to_dict()

    def pop_activity_events(self) -> list[dict[str, Any]]:
        """Legacy destructive-looking API backed by a shared event journal.

        Old TUI clients retain their one-shot polling behaviour while Desktop
        clients can use a cursor without stealing events from the TUI.
        """
        with self._event_lock:
            self._collect_activity_events()
            events = [event for event in self._activity_event_journal if int(event["sequence"]) > self._legacy_activity_cursor]
            if events:
                self._legacy_activity_cursor = int(events[-1]["sequence"])
            return [_without_sequence(event) for event in events]

    def activity_events_since(
        self,
        *,
        after_sequence: int = 0,
        integration_id: str = "",
        channel_target: str = "",
        limit: int = 100,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return a non-destructive, client-cursored view of activity."""
        with self._event_lock:
            self._collect_activity_events()
            cursor = max(0, int(after_sequence or 0))
            matched = [
                event
                for event in self._activity_event_journal
                if int(event["sequence"]) > cursor
                and (not integration_id or event["integration_id"] == integration_id)
                and (not channel_target or event["channel_target"] == channel_target)
            ][: max(1, min(int(limit or 100), 500))]
            next_sequence = int(matched[-1]["sequence"]) if matched else cursor
            return matched, next_sequence

    def poll_desktop(
        self,
        *,
        client_id: str,
        user: str,
        project_id: str = "",
        after_sequence: int = 0,
        after_ui_sequence: int = 0,
        client_capabilities: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Collect Desktop activity and atomically lease its pending messages."""
        normalized_user = self._admin_user_id(user)
        normalized_project = str(project_id or "").strip()
        normalized_client = str(client_id or "desktop").strip() or "desktop"
        capabilities = dict(client_capabilities or {})
        catalogs = capabilities.get("supportedCatalogIds")
        self._desktop_capabilities[normalized_client] = {
            str(catalog).strip() for catalog in catalogs if str(catalog).strip()
        } if isinstance(catalogs, list) else set()
        events, next_sequence = self.activity_events_since(
            after_sequence=after_sequence,
            integration_id="desktop",
            channel_target=normalized_user,
            limit=limit,
        )
        selector = OutboundSelector(integration_id="desktop", channel_target=normalized_user, status="pending")
        deliveries: list[dict[str, Any]] = []
        for _ in range(max(1, min(int(limit or 20), 100))):
            delivery = self.runtime.outbox.claim_next(selector, lease_owner=f"desktop:{normalized_client}", lease_seconds=120)
            if delivery is None:
                break
            deliveries.append({
                **delivery.to_dict(),
                "conversation_sequence": self.runtime.conversation_store.sequence_for_source_message_id(
                    f"outbound:{delivery.outbox_message_id}"
                ),
            })
        questions = []
        for question in self.runtime.question_store.list_pending_for_respondent(
            normalized_user,
            project_id=normalized_project,
        ):
            event = self._record_question_conversation_event(question)
            questions.append({
                **question.to_dict(),
                "conversation_sequence": event.sequence if event is not None else self.runtime.conversation_store.sequence_for_source_message_id(
                    f"question:{question.question_id}"
                ),
            })
        ui_events, next_ui_sequence = self.ui_events_since(
            after_sequence=after_ui_sequence,
            user=normalized_user,
            limit=limit,
        )
        if ALPHONSE_DESKTOP_CATALOG_ID in self._desktop_capabilities[normalized_client]:
            ui_events = self._a2ui_task_progress_events(
                events,
                client_id=normalized_client,
                user=normalized_user,
                project_id=normalized_project,
            ) + ui_events
            ui_events = self._a2ui_scheduled_task_events(ui_events, project_id=normalized_project)
            ui_events.extend(
                self._sync_question_surfaces(
                    client_id=normalized_client,
                    user=normalized_user,
                    project_id=normalized_project,
                )
            )
        else:
            ui_events = [item for item in ui_events if _event_name(item) != "scheduled_task_created"]
        return {
            "events": events,
            "next_sequence": next_sequence,
            "deliveries": deliveries,
            "questions": questions,
            "ui_events": ui_events,
            "next_ui_sequence": next_ui_sequence,
            "server_capabilities": self._a2ui.server_capabilities(),
            "project_attention": self._desktop_project_attention(normalized_user),
            "status": {"active_work": self.active_work(), "activity": self.activity_status(), "queue": self._inbound_queue_status()},
        }

    def acknowledge_desktop_delivery(self, *, client_id: str, outbox_message_id: str) -> bool:
        delivery = self.runtime.outbox.get(outbox_message_id)
        expected_owner = f"desktop:{str(client_id or 'desktop').strip() or 'desktop'}"
        if delivery is None or delivery.integration_id != "desktop" or delivery.lease_owner != expected_owner:
            return False
        acknowledged = self.runtime.outbox.mark_delivered(outbox_message_id)
        if acknowledged and delivery.task_id:
            key = (str(client_id or "desktop").strip() or "desktop", str(delivery.audience_user_id or "").strip())
            self._desktop_progress_closures.setdefault(key, set()).add(str(delivery.task_id))
        return acknowledged

    def desktop_conversation_history(self, *, user: str, project_id: str = "", limit: int = 100) -> list[dict[str, Any]]:
        normalized_user = self._admin_user_id(user)
        normalized_project = str(project_id or "").strip()
        for question in self.runtime.question_store.list_for_conversation(
            normalized_user,
            project_id=normalized_project,
        ):
            self._record_question_conversation_event(question)
        timeline = self.runtime.conversation_store.list(owner_user_id=normalized_user, project_id=normalized_project, limit=limit)
        if not timeline and not self.runtime.conversation_store.legacy_import_completed(
            owner_user_id=normalized_user,
            project_id=normalized_project,
        ):
            legacy = self.runtime.core.memory.latest_content(user_id=normalized_user, project_id=normalized_project)
            recovered = legacy_ledger_events(
                legacy,
                owner_user_id=normalized_user,
                project_id=normalized_project,
                limit=limit,
            )
            self.runtime.conversation_store.import_legacy_events(
                owner_user_id=normalized_user,
                project_id=normalized_project,
                events=recovered,
            )
            timeline = self.runtime.conversation_store.list(
                owner_user_id=normalized_user,
                project_id=normalized_project,
                limit=limit,
            )
        return [
            {
                "id": _conversation_message_id(event.source_message_id, event.event_id),
                "role": event.role,
                "content": event.content,
                "source": event.source,
                "created_at": event.created_at,
                "project_id": event.project_id,
                "sequence": event.sequence,
            }
            for event in timeline
        ]

    def mark_desktop_project_seen(self, *, user: str, project_id: str, through_sequence: int | None = None) -> dict[str, Any]:
        normalized_user = self._admin_user_id(user)
        normalized_project = str(project_id or "").strip()
        sequence = self.runtime.conversation_store.mark_project_seen(
            owner_user_id=normalized_user,
            project_id=normalized_project,
            through_sequence=through_sequence,
        )
        return {"project_id": normalized_project, "seen_through_sequence": sequence}

    def _desktop_project_attention(self, user: str) -> dict[str, dict[str, int]]:
        unread = self.runtime.conversation_store.project_unread_counts(owner_user_id=user)
        questions = self.runtime.question_store.pending_counts_by_project(user)
        return {
            project_id: {
                "unread_messages": int(unread.get(project_id, 0)),
                "pending_questions": int(questions.get(project_id, 0)),
                "total": int(unread.get(project_id, 0)) + int(questions.get(project_id, 0)),
            }
            for project_id in sorted(set(unread) | set(questions))
        }

    def list_projects(self, *, user: str) -> list[dict[str, str]]:
        normalized = self._admin_user_id(user)
        return [project.to_dict() for project in self.runtime.project_store.list_visible_projects(normalized, requester_is_admin=True)]

    def project_recent_files(self, *, user: str, project_id: str, limit: int = 4) -> list[dict[str, str]]:
        """Return the newest accessible direct children of an authorized project root."""
        actor = self._admin_user_id(user)
        project = self.runtime.project_store.get_project(
            project_id,
            requester_user_id=actor,
            requester_is_admin=True,
        )
        if project is None:
            raise ValueError("project_not_found")

        root = Path(project.root_path).expanduser()
        if not root.is_dir():
            return []
        entries: list[tuple[float, dict[str, str]]] = []
        try:
            children = root.iterdir()
            for child in children:
                if child.name.startswith(".") or not os.access(child, os.R_OK):
                    continue
                try:
                    stat = child.stat()
                    kind = "directory" if child.is_dir() else "file"
                except OSError:
                    continue
                entries.append((stat.st_mtime, {
                    "name": child.name,
                    "kind": kind,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                }))
        except OSError:
            return []
        entries.sort(key=lambda item: item[0], reverse=True)
        bounded_limit = max(1, min(int(limit or 4), 4))
        return [entry for _, entry in entries[:bounded_limit]]

    def copy_desktop_project_files(self, *, user: str, project_id: str, source_paths: list[str]) -> list[dict[str, Any]]:
        """Copy user-selected Desktop files into an authorized project root at send time."""
        actor = self._admin_user_id(user)
        project = self.runtime.project_store.get_project(
            str(project_id or "").strip(), requester_user_id=actor, requester_is_admin=True,
        )
        if project is None:
            raise ValueError("project_required")
        root = Path(project.root_path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("project_directory_unavailable")
        paths = [str(item or "").strip() for item in source_paths]
        if not paths:
            return []

        max_bytes = 50 * 1024 * 1024
        sources: list[tuple[Path, os.stat_result]] = []
        for raw_path in paths:
            source = Path(raw_path).expanduser()
            if not raw_path or source.is_symlink() or not source.is_file() or not os.access(source, os.R_OK):
                raise ValueError("desktop_attachment_file_unavailable")
            try:
                stat = source.stat()
            except OSError as exc:
                raise ValueError("desktop_attachment_file_unavailable") from exc
            if stat.st_size > max_bytes:
                raise ValueError("desktop_attachment_too_large")
            sources.append((source, stat))

        created: list[Path] = []
        registered_assets: list[str] = []
        descriptors: list[dict[str, Any]] = []
        try:
            for source, stat in sources:
                destination = self._copy_desktop_file_to_project(source, root)
                created.append(destination)
                mime_type = mimetypes.guess_type(destination.name)[0] or "application/octet-stream"
                asset_id = ""
                if mime_type.startswith("image/"):
                    asset = self.runtime.asset_store.register_bytes(
                        owner_user_id=actor,
                        descriptor=AttachmentDescriptor(destination.name, mime_type, stat.st_size),
                        content=destination.read_bytes(),
                        source="desktop",
                    )
                    asset_id = asset.asset_id
                    registered_assets.append(asset_id)
                descriptors.append({
                    "asset_id": asset_id,
                    "filename": destination.name,
                    "mime_type": mime_type,
                    "size_bytes": stat.st_size,
                    "kind": "desktop_project_file",
                    "ingestion_status": "copied",
                    "project_path": str(destination),
                    "relative_path": destination.name,
                    "caption": "",
                })
        except Exception:
            for asset_id in registered_assets:
                self.runtime.asset_store.delete(asset_id, requester_user_id=actor)
            for destination in created:
                try:
                    destination.unlink()
                except OSError:
                    pass
            raise
        return descriptors

    @staticmethod
    def _copy_desktop_file_to_project(source: Path, root: Path) -> Path:
        safe_name = source.name[:180] or "attachment"
        stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
        temporary = root / f".alphonse-upload-{uuid4().hex}.tmp"
        shutil.copy2(source, temporary)
        try:
            for index in range(10_000):
                candidate_name = safe_name if index == 0 else f"{stem} ({index}){suffix}"
                destination = root / candidate_name
                try:
                    os.link(temporary, destination)
                except FileExistsError:
                    continue
                return destination
            raise RuntimeError("desktop_attachment_name_conflict")
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass

    def manageable_projects(self, *, user: str, status: str = "") -> list[dict[str, Any]]:
        actor = self._admin_user_id(user)
        status_value = str(status or "").strip() or None
        projects = self.runtime.project_store.list_manageable_projects(actor, requester_is_admin=True, status=status_value)  # type: ignore[arg-type]
        users = {item["user_id"]: item for item in self.list_users()}
        return [{**project.to_dict(), "owner": users.get(project.owner_user_id)} for project in projects]

    def create_project(self, *, user: str, name: str, description: str, root_path: str, visibility: str) -> dict[str, str]:
        owner = self._admin_user_id(user)
        from pathlib import Path
        slug = "-".join(part for part in "".join(char.lower() if char.isalnum() else " " for char in name).split()) or "project"
        parent = Path(root_path).expanduser() if str(root_path).strip() else self.runtime.user_store.managed_project_root(owner)
        project = self.runtime.project_store.create_project(
            name=name,
            description=description,
            root_path=str(parent / slug),
            visibility=visibility,  # type: ignore[arg-type]
            owner_user_id=owner,
        )
        return project.to_dict()

    def import_project(self, *, user: str, name: str, description: str, root_path: str, visibility: str) -> dict[str, Any]:
        owner = self._admin_user_id(user)
        root = Path(root_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError("project_import_directory_required")
        if self.runtime.project_store.find_project_by_root(str(root)) is not None:
            raise ValueError("project_root_already_registered")
        return self.runtime.project_store.create_project(name=name, description=description, root_path=str(root), visibility=visibility, owner_user_id=owner).to_dict()  # type: ignore[arg-type]

    def update_project(self, *, user: str, project_id: str, name: str, description: str, visibility: str) -> dict[str, Any]:
        actor = self._admin_user_id(user)
        return self.runtime.project_store.update_project(project_id, name=name, description=description, visibility=visibility, requester_user_id=actor, requester_is_admin=True).to_dict()  # type: ignore[arg-type]

    def archive_project(self, *, user: str, project_id: str) -> dict[str, Any]:
        actor = self._admin_user_id(user)
        self._ensure_project_has_no_live_schedules(project_id)
        project = self.runtime.project_store.archive_project(project_id, requester_user_id=actor, requester_is_admin=True)
        self.runtime.project_session_store.clear_project(project.project_id)
        if self.runtime.active_project_id == project.project_id: self.runtime.active_project_id = ""
        return project.to_dict()

    def restore_project(self, *, user: str, project_id: str) -> dict[str, Any]:
        actor = self._admin_user_id(user)
        return self.runtime.project_store.restore_project(project_id, requester_user_id=actor, requester_is_admin=True).to_dict()

    def delete_project(self, *, user: str, project_id: str, confirmation: str) -> dict[str, Any]:
        actor = self._admin_user_id(user)
        if str(confirmation or "") != str(project_id or ""):
            raise ValueError("delete_confirmation_must_match_project_id")
        self._ensure_project_has_no_live_schedules(project_id)
        project = self.runtime.project_store.delete_project(project_id, requester_user_id=actor, requester_is_admin=True)
        self.runtime.project_session_store.clear_project(project.project_id)
        if self.runtime.active_project_id == project.project_id: self.runtime.active_project_id = ""
        root = Path(project.root_path).resolve()
        managed = self.runtime.user_store.managed_project_root(project.owner_user_id).resolve()
        removed_files = False
        try:
            root.relative_to(managed)
            if root.exists():
                shutil.rmtree(root)
                removed_files = True
        except ValueError:
            pass
        return {"deleted": True, "project_id": project.project_id, "removed_managed_files": removed_files}

    def _ensure_project_has_no_live_schedules(self, project_id: str) -> None:
        tasks = self.runtime.schedule_store.list_tasks(project_id=str(project_id or ""), limit=1000)
        if any(task.status in {"active", "paused"} for task in tasks):
            raise ValueError("project_has_active_scheduled_tasks")

    def project_members(self, project_id: str) -> list[str]:
        self._admin_user_id()
        return self.runtime.project_store.list_members(project_id)

    def add_project_member(self, project_id: str, user_id: str) -> None:
        self._admin_user_id()
        if self.runtime.user_store.get_user(user_id) is None:
            raise KeyError("user_not_found")
        self.runtime.project_store.add_member(project_id, user_id)

    def remove_project_member(self, project_id: str, user_id: str) -> bool:
        self._admin_user_id()
        return self.runtime.project_store.remove_member(project_id, user_id)

    def select_project_session(
        self,
        *,
        user: str,
        integration_id: str,
        channel_target: str,
        thread_id: str,
        project_id: str,
    ) -> dict[str, str]:
        user = self._admin_user_id(user)
        key = ProjectSessionKey(user, integration_id, channel_target or user, thread_id)
        project = self.runtime.inbound_router.select_project(key, project_id)
        return self.runtime.project_session_store.set(key, project).to_dict()

    def active_project_session(self, *, user: str, integration_id: str, channel_target: str, thread_id: str) -> dict[str, str] | None:
        user = self._admin_user_id(user)
        key = ProjectSessionKey(user, integration_id, channel_target or user, thread_id)
        project = self.runtime.inbound_router.active_project(key)
        if project is None:
            return None
        session = self.runtime.project_session_store.get(key)
        return session.to_dict() if session is not None else None

    def ingest_message(self, **values: Any) -> dict[str, Any]:
        routed = self.runtime.inbound_router.ingest(**values)
        return {
            "message_id": routed.queued.message_id if routed.queued is not None else "",
            "handled_command": routed.handled_command,
            "project_id": routed.project_id,
            "routing_disposition": routed.disposition,
            "turns_ahead": routed.turns_ahead,
            "created_at": routed.queued.message.timestamp.isoformat() if routed.queued is not None else "",
            "conversation_sequence": self.runtime.conversation_store.sequence_for_source_message_id(
                f"inbound:{routed.queued.message_id}"
            ) if routed.queued is not None else 0,
        }

    def publish_event(self, **values: Any) -> dict[str, Any]:
        assert self.event_store is not None
        result = self.event_store.publish(
            worker_id=str(values.get("worker_id") or ""), event_id=str(values.get("event_id") or ""),
            event_type=str(values.get("event_type") or ""), event_version=str(values.get("event_version") or ""),
            occurred_at=str(values.get("occurred_at") or ""), payload=dict(values.get("payload") or {}),
        )
        if not result.get("accepted") or result.get("duplicate"):
            return result
        for execution in self.event_store.claim_event_executions():
            payload = _event_payload(execution)
            queued = self.runtime.channel.queue_message(
                prompt=str(execution["prompt"]), user=str(execution["owner_user_id"]), project_id=str(execution["project_id"]),
                metadata={"source": "event_automation", "automation_id": str(execution["automation_id"]), "automation_execution_id": str(execution["execution_id"]), "event": payload, "channel": _json_object(execution.get("origin_channel_json"))},
                message_id=f"event:{execution['execution_id']}",
            )
            self.event_store.mark_execution_enqueued(str(execution["execution_id"]), queued.message_id)
        return result

    def register_event_worker(self, **values: Any) -> dict[str, Any]:
        assert self.event_store is not None
        return self.event_store.register_worker(worker_id=str(values.get("worker_id") or ""), display_name=str(values.get("display_name") or ""), allowed_event_types=[str(item) for item in values.get("allowed_event_types", []) if str(item)], enabled=bool(values.get("enabled", True))).to_dict()

    def register_event_type(self, **values: Any) -> dict[str, Any]:
        assert self.event_store is not None
        return self.event_store.register_event_type(event_type=str(values.get("event_type") or ""), version=str(values.get("version") or ""), schema=dict(values.get("schema") or {}), max_history=int(values.get("max_history") or 500), enabled=bool(values.get("enabled", True))).to_dict()

    def create_event_automation(self, **values: Any) -> dict[str, Any]:
        assert self.event_store is not None
        owner = self._admin_user_id(str(values.get("owner_user_id") or values.get("user") or ""))
        project_id = str(values.get("project_id") or "").strip()
        project = self.runtime.project_store.get_project(project_id, requester_user_id=owner, requester_is_admin=self.runtime.user_store.is_admin(owner))
        if project is None:
            raise ValueError("automation_project_not_accessible")
        return self.event_store.create_event_automation(owner_user_id=owner, name=str(values.get("name") or ""), prompt=str(values.get("prompt") or ""), event_type=str(values.get("event_type") or ""), event_version=str(values.get("event_version") or ""), filters=dict(values.get("filters") or {}), project_id=project.project_id, origin_channel=dict(values.get("origin_channel") or {}), enabled=bool(values.get("enabled", True))).to_dict()

    def automation_catalog(self) -> dict[str, Any]:
        assert self.event_store is not None
        schedules = [
            {"automation_id": task.scheduled_task_id, "name": task.name, "trigger_kind": "schedule", "status": task.status, "trigger": dict(task.schedule)}
            for task in self.runtime.schedule_store.list_tasks(limit=1000)
        ]
        events = [{**item.to_dict(), "trigger_kind": "event"} for item in self.event_store.list_automations()]
        return {"workers": [item.to_dict() for item in self.event_store.list_workers()], "event_types": [item.to_dict() for item in self.event_store.list_event_types()], "automations": schedules + events, "events": self.event_store.list_events(limit=100)}

    def read_project_context(self, *, user: str, project_id: str) -> dict[str, str]:
        user = self._admin_user_id(user)
        project = self.runtime.project_store.get_project(project_id, requester_user_id=user, requester_is_admin=True)
        if project is None:
            raise KeyError(f"project_not_found: {project_id}")
        return {"project_id": project.project_id, "content": self.runtime.project_store.read_project_context(project.project_id, requester_user_id=user)}

    def save_project_context(self, *, user: str, project_id: str, content: str) -> dict[str, str]:
        return self.runtime.project_store.write_project_context(project_id, content, requester_user_id=self._admin_user_id(user), requester_is_admin=True).to_dict()

    def answer_question(self, *, user: str, question_id: str, text: str = "", payload: dict[str, Any] | None = None) -> dict[str, Any]:
        normalized_user = self._admin_user_id(user)
        question = self.runtime.question_store.get_question(question_id)
        if question is not None:
            self._record_question_conversation_event(question)
        if question is not None and question.kind == "datetime" and isinstance(payload, dict):
            raw = str(payload.get("datetime") or "").strip()
            try:
                parsed = datetime.fromisoformat(raw)
                if parsed.tzinfo is None:
                    payload = {**payload, "datetime": _local_datetime_to_utc(parsed, timezone_name=self.runtime.user_store.timezone())}
            except (ValueError, TypeError):
                pass
        result = self.runtime.question_store.route_answer(
            respondent_user_id=normalized_user,
            question_id=question_id,
            text=text or None,
            payload=payload,
        )
        if result.handled and result.question is not None:
            child_id = str(result.question.metadata.get("child_task_id") or "").strip()
            if child_id:
                child = self.runtime.question_store.load_task_checkpoint(child_id)
                if child is not None:
                    self.runtime.core.memory.event(child, "Conversation", f"- {normalized_user}: {text or _question_answer_text(payload)}")
                    child.status = "completed"
                    child.outcome = {"status": "success", "answered_question_id": result.question.question_id}
                    self.runtime.core.memory.finish_task(child)
                    self.runtime.question_store.mark_task_checkpoint_terminal(child_id, status="done")
        if result.handled and result.resumed_task is not None:
            self.runtime.ui_events.append(
                CoreUiEvent(
                    event_type="question_interrupt_resolved",
                    payload={"question": result.question.to_dict() if result.question else None, "answer": result.answer},
                )
            )
            queued = self.runtime.channel.queue_message(
                prompt=text or _question_answer_text(payload),
                user=normalized_user,
                project_id=result.resumed_task.project_id,
                correlation_id=result.resumed_task.correlation_id,
                metadata={
                    "task_state": result.resumed_task.to_dict(),
                    "answered_question_id": result.question.question_id if result.question else "",
                    "routing_disposition": "correlated_response",
                },
                integration_id="desktop",
                provider_key="tui",
                channel_target=normalized_user,
            )
            payload_result = result.to_dict()
            payload_result["message_id"] = queued.message_id
            return payload_result
        return result.to_dict()

    def _record_question_conversation_event(self, question: Any) -> Any:
        """Make an interrupt question a durable assistant turn exactly once."""
        return self.runtime.conversation_store.record(
            owner_user_id=str(question.respondent_user_id or "").strip(),
            project_id=str(question.project_id or "").strip(),
            role="assistant",
            content=str(question.message or "").strip(),
            source="desktop",
            source_message_id=f"question:{str(question.question_id or '').strip()}",
            created_at=str(question.created_at or ""),
        )

    def cancel_question(self, question_id: str) -> bool:
        question = self.runtime.question_store.get_question(question_id)
        cancelled = self.runtime.question_store.cancel_question(question_id)
        if cancelled:
            self.runtime.ui_events.append(
                CoreUiEvent(
                    event_type="question_interrupt_cancelled",
                    payload={"question": question.to_dict() if question else None, "cancelled": True},
                )
            )
        return cancelled

    def a2ui_action(
        self,
        *,
        client_id: str,
        user: str,
        surface_id: str,
        source_component_id: str,
        action_name: str,
        context: dict[str, Any] | None = None,
        data_model: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform a validated action from a server-owned A2UI surface."""
        client = str(client_id or "desktop").strip() or "desktop"
        respondent = self._admin_user_id(user)
        if ALPHONSE_DESKTOP_CATALOG_ID not in self._desktop_capabilities.get(client, set()):
            raise ValueError("a2ui_catalog_not_negotiated")
        values = dict(context or {})
        action = str(action_name or "").strip()
        source = str(source_component_id or "").strip()
        scheduled_task_id = _scheduled_task_id_from_surface(surface_id)
        if action == "view_scheduled_task":
            if source != "view" or not scheduled_task_id or scheduled_task_id != str(values.get("scheduled_task_id") or ""):
                raise ValueError("a2ui_surface_or_context_invalid")
            task = self._scheduled_task_for_actor(actor_user_id=respondent, scheduled_task_id=scheduled_task_id)
            return {
                "action": "view_scheduled_task",
                "scheduled_task_id": scheduled_task_id,
                "project_id": task.project_id,
            }
        question_id = question_id_from_surface(surface_id)
        if not question_id or question_id != str(values.get("question_id") or ""):
            raise ValueError("a2ui_surface_or_context_invalid")
        question = self.runtime.question_store.get_question(question_id)
        if question is None or question.status != "pending" or question.respondent_user_id != respondent:
            raise ValueError("a2ui_question_not_available")
        if action == "cancel_question" and source == "cancel":
            return {"cancelled": self.cancel_question(question_id)}
        if action != "answer_question":
            raise ValueError("a2ui_action_not_allowed")
        payload: dict[str, Any]
        text = ""
        if question.kind == "yes_no":
            model = dict(data_model or {})
            answer = model.get("answer") if isinstance(model.get("answer"), dict) else {}
            if source != "submit" or not isinstance(answer.get("answer"), bool):
                raise ValueError("a2ui_answer_invalid")
            payload, text = {"answer": answer["answer"]}, "yes" if answer["answer"] else "no"
        elif question.kind in {"single_choice", "multi_choice"}:
            model = dict(data_model or {})
            answer = model.get("answer") if isinstance(model.get("answer"), dict) else {}
            selected = answer.get("choice_ids") if isinstance(answer.get("choice_ids"), list) else []
            selected_ids = [str(item) for item in selected]
            valid = {choice.id: choice.label for choice in question.choices}
            if source != "submit" or not selected_ids or len(selected_ids) != len(set(selected_ids)) or any(choice_id not in valid for choice_id in selected_ids):
                raise ValueError("a2ui_choice_invalid")
            if question.kind == "single_choice" and len(selected_ids) != 1:
                raise ValueError("a2ui_choice_invalid")
            payload = {"choice_id": selected_ids[0]} if question.kind == "single_choice" else {"choice_ids": selected_ids}
            text = ", ".join(valid[choice_id] for choice_id in selected_ids)
        elif question.kind == "datetime":
            model = dict(data_model or {})
            answer = model.get("answer") if isinstance(model.get("answer"), dict) else {}
            raw = str(answer.get("datetime") or "").strip()
            try:
                local = datetime.fromisoformat(raw)
                if local.tzinfo is not None:
                    raise ValueError
                converted = _local_datetime_to_utc(local, timezone_name=self.runtime.user_store.timezone())
            except (ValueError, TypeError):
                raise ValueError("a2ui_datetime_invalid") from None
            if source != "submit":
                raise ValueError("a2ui_datetime_invalid")
            payload, text = {"datetime": converted}, converted
        else:
            model = dict(data_model or {})
            answer = model.get("answer") if isinstance(model.get("answer"), dict) else {}
            text = str(answer.get("text") or "").strip()
            payload = {"text": text}
            if source != "submit" or not text:
                raise ValueError("a2ui_text_invalid")
        return self.answer_question(user=respondent, question_id=question_id, text=text, payload=payload)

    def _a2ui_scheduled_task_events(self, events: list[dict[str, Any]], *, project_id: str) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        for item in events:
            event = item.get("event") if isinstance(item.get("event"), dict) else {}
            if event.get("type") != "CUSTOM" or event.get("name") != "scheduled_task_created":
                rendered.append(item)
                continue
            payload = event.get("value") if isinstance(event.get("value"), dict) else {}
            task = payload.get("scheduled_task") if isinstance(payload.get("scheduled_task"), dict) else {}
            if str(task.get("project_id") or "").strip() != project_id:
                continue
            try:
                rendered.extend({"event": _a2ui_custom(envelope)} for envelope in self._a2ui.scheduled_task_created(task, project_name=str(payload.get("project_name") or "")))
            except ValueError:
                continue
        return rendered

    def list_integrations(self) -> list[dict[str, Any]]:
        records = {record.provider_key: record for record in self.runtime.integration_store.list()}
        return [
            {
                "provider_key": descriptor.provider_key,
                "display_name": descriptor.display_name,
                "integration": records[descriptor.provider_key].to_dict() if descriptor.provider_key in records else None,
            }
            for descriptor in self.runtime.integration_registry.list()
        ]

    def save_telegram_integration(self, *, user: str, values: dict[str, Any]) -> dict[str, Any]:
        existing = self.runtime.integration_store.get(str(values.get("integration_id") or "")) or self.runtime.integration_store.get_by_provider("telegram")
        secrets = dict(existing.secrets) if existing is not None else {}
        token = str(values.get("bot_token") or "").strip()
        if bool(values.get("remove_token")):
            secrets.pop("bot_token", None)
        elif token:
            secrets["bot_token"] = token
        enabled = bool(values.get("enabled"))
        if enabled and not str(secrets.get("bot_token") or "").strip():
            raise ValueError("telegram_bot_token_required")
        provider_user_id = str(values.get("telegram_user_id") or "").strip()
        if provider_user_id:
            self.runtime.user_store.bind_address(
                user_id=self._admin_user_id(user),
                integration_id=str(values.get("integration_id") or "telegram-home"),
                provider_key="telegram",
                provider_user_id=provider_user_id,
            )
        record = self.runtime.integration_store.upsert(
            integration_id=str(values.get("integration_id") or "telegram-home").strip() or "telegram-home",
            provider_key="telegram",
            display_name=str(values.get("display_name") or "Telegram").strip() or "Telegram",
            enabled=enabled,
            config={
                "poll_interval_sec": _positive_float(values.get("poll_interval_sec")),
                "allowed_chat_ids": _comma_values(values.get("allowed_chat_ids")),
                "owner_user_id": self._admin_user_id(user),
                "telegram_user_id": provider_user_id,
                "presence_enabled": bool(values.get("presence_enabled", True)),
            },
            secrets=secrets,
        )
        refresh_runtime_identity_resolver(self.runtime)
        self.restart_integrations()
        return record.to_dict()

    def save_discord_integration(self, *, user: str, values: dict[str, Any]) -> dict[str, Any]:
        existing = self.runtime.integration_store.get(str(values.get("integration_id") or "")) or self.runtime.integration_store.get_by_provider("discord")
        secrets = dict(existing.secrets) if existing is not None else {}
        token = str(values.get("bot_token") or "").strip()
        if bool(values.get("remove_token")):
            secrets.pop("bot_token", None)
        elif token:
            secrets["bot_token"] = token
        enabled = bool(values.get("enabled"))
        if enabled and not str(secrets.get("bot_token") or "").strip():
            raise ValueError("discord_bot_token_required")
        integration_id = str(values.get("integration_id") or "discord-home").strip() or "discord-home"
        provider_user_id = str(values.get("discord_user_id") or "").strip()
        if provider_user_id:
            self.runtime.user_store.bind_address(
                user_id=self._admin_user_id(user), integration_id=integration_id,
                provider_key="discord", provider_user_id=provider_user_id,
            )
        record = self.runtime.integration_store.upsert(
            integration_id=integration_id,
            provider_key="discord",
            display_name=str(values.get("display_name") or "Discord").strip() or "Discord",
            enabled=enabled,
            config={
                "allowed_guild_ids": _comma_values(values.get("allowed_guild_ids")),
                "allowed_channel_ids": _comma_values(values.get("allowed_channel_ids")),
                "owner_user_id": self._admin_user_id(user),
                "discord_user_id": provider_user_id,
                "presence_enabled": bool(values.get("presence_enabled", True)),
            },
            secrets=secrets,
        )
        refresh_runtime_identity_resolver(self.runtime)
        self.restart_integrations()
        return record.to_dict()

    def _collect_activity_events(self) -> None:
        events = list(self.runtime.activity_events)
        self.runtime.activity_events.clear()
        for event in events:
            self._activity_event_sequence += 1
            self._activity_event_journal.append(
                {
                    "sequence": self._activity_event_sequence,
                    "phase": event.phase.value,
                    "label": event.label,
                    "message": event.message,
                    "speaker": event.speaker,
                    "task_id": event.task_id,
                    "message_id": event.message_id,
                    "user": event.user,
                    "integration_id": event.integration_id,
                    "channel_target": event.channel_target,
                    "progress": dict(event.progress),
                }
            )
        if len(self._activity_event_journal) > 2000:
            self._activity_event_journal = self._activity_event_journal[-2000:]

    def _a2ui_task_progress_events(
        self,
        events: list[dict[str, Any]],
        *,
        client_id: str,
        user: str,
        project_id: str,
    ) -> list[dict[str, Any]]:
        """Emit task-progress cards only for the local admin's Desktop work."""
        admin = self.runtime.user_store.admin_user()
        if admin is None or user != admin.user_id:
            return []
        key = (client_id, user)
        known = self._desktop_progress_surfaces.setdefault(key, set())
        closing = self._desktop_progress_closures.pop(key, set())
        rendered = [
            {"event": _a2ui_custom(self._a2ui.task_progress_closed(task_id))}
            for task_id in sorted(closing)
        ]
        known.difference_update(closing)
        for event in events:
            if event.get("user") != admin.user_id or event.get("integration_id") != "desktop":
                continue
            task_id = str(event.get("task_id") or "").strip()
            progress = event.get("progress") if isinstance(event.get("progress"), dict) else {}
            if not task_id or not progress or str(progress.get("project_id") or "").strip() != project_id:
                continue
            payload = {
                **progress,
                "phase": str(event.get("phase") or "working"),
                "label": str(event.get("label") or "Working"),
                "message": str(event.get("message") or ""),
            }
            rendered.extend({"event": _a2ui_custom(envelope)} for envelope in self._a2ui.task_progress(task_id, payload))
            known.add(task_id)
        return rendered

    def _inbound_queue_status(self) -> dict[str, int]:
        counts = getattr(self.runtime.queue, "status_counts", lambda: {})()
        values = dict(counts) if isinstance(counts, dict) else {}
        ready = int(self.runtime.queue.size() or 0)
        processing = int(values.get("processing", 0) or 0)
        return {"ready": ready, "processing": processing}

    def ui_events_since(self, *, after_sequence: int = 0, user: str, limit: int = 100) -> tuple[list[dict[str, Any]], int]:
        """Return ordered AG-UI events without changing TUI activity polling."""
        with self._ui_event_lock:
            self._collect_ui_events()
            cursor = max(0, int(after_sequence or 0))
            matched = [
                event
                for event in self._ui_event_journal
                if int(event["sequence"]) > cursor and (not event["user"] or event["user"] == user)
            ][: max(1, min(int(limit or 100), 500))]
            return matched, int(matched[-1]["sequence"]) if matched else cursor

    def _collect_ui_events(self) -> None:
        events = list(self.runtime.ui_events)
        self.runtime.ui_events.clear()
        for core_event in events:
            for event in self._ag_ui.map_event(core_event):
                self._ui_event_sequence += 1
                self._ui_event_journal.append(
                    {
                        "sequence": self._ui_event_sequence,
                        "user": _ui_event_user(core_event),
                        "event": event,
                    }
                )
        if len(self._ui_event_journal) > 2000:
            self._ui_event_journal = self._ui_event_journal[-2000:]

    def _sync_question_surfaces(self, *, client_id: str, user: str, project_id: str) -> list[dict[str, Any]]:
        """Reconcile trusted question surfaces per Desktop client.

        This is deliberately state based: a reconnect (and question expiry) is
        recoverable even if a prior poll response was lost.
        """
        key = (client_id, user, project_id)
        known = self._desktop_surfaces.setdefault(key, set())
        pending = self.runtime.question_store.list_pending_for_respondent(user, project_id=project_id)
        expected = {surface_id_for_question(question.question_id): question for question in pending}
        events: list[dict[str, Any]] = []
        for surface_id in sorted(known - set(expected)):
            events.append({"event": _a2ui_custom(self._a2ui.question_closed(question_id_from_surface(surface_id)))})
        for surface_id, question in expected.items():
            if surface_id not in known:
                events.extend({"event": _a2ui_custom(message)} for message in self._a2ui.question_opened(question, timezone=self.runtime.user_store.timezone()))
        self._desktop_surfaces[key] = set(expected)
        return events

    def active_work(self) -> dict[str, str]:
        with self._active_work_lock:
            return dict(self._active_work)

    def activity_status(self) -> dict[str, str]:
        with self._active_work_lock:
            return dict(self._activity_status)

    def run_once(self) -> Any:
        self._run_retention_if_due()
        queued = self.runtime.queue.peek()
        self._set_active_work(queued)
        if queued is not None:
            self._set_activity_status("working")
        with self.runtime.presence_projector.processing(queued):
            step = self.runtime.core.step()
            if step.status in {
                LoopStepStatus.PROCESSED,
                LoopStepStatus.PARKED,
                LoopStepStatus.WAITING,
                LoopStepStatus.CANCELLED,
                LoopStepStatus.FAILED,
            }:
                self.runtime.presence_projector.finish(
                    failed=step.status == LoopStepStatus.FAILED,
                    waiting=step.status in {LoopStepStatus.PARKED, LoopStepStatus.WAITING},
                )
        queued_metadata = getattr(getattr(queued, "message", None), "metadata", {}) or {}
        occurrence_key = str(queued_metadata.get("occurrence_key") or "").strip() if isinstance(queued_metadata, dict) and str(queued_metadata.get("source") or "") == "scheduled_task" else ""
        if occurrence_key and step.queued_message_id:
            self.runtime.schedule_store.mark_occurrence_processing(occurrence_key)
        snapshot = self.runtime.visible_state.snapshot()
        if step.status in {LoopStepStatus.PROCESSED, LoopStepStatus.PARKED, LoopStepStatus.WAITING}:
            outbox_path = str(getattr(self.runtime.outbox, "db_path", ":memory:"))
            conversation_path = str(getattr(self.runtime.conversation_store, "db_path", ":memory:"))
            schedule_path = str(getattr(self.runtime.schedule_store, "db_path", ":memory:"))
            if outbox_path != ":memory:" and outbox_path == conversation_path == schedule_path:
                with transaction(outbox_path, immediate=True) as connection:
                    projected = self._project_snapshot_delivery(snapshot, connection=connection)
                    self._mark_projected_occurrence(projected, connection=connection)
            else:
                projected = self._project_snapshot_delivery(snapshot)
                self._mark_projected_occurrence(projected)
            self._emit_scheduled_task_card(snapshot)
            if step.queued_message_id:
                acknowledge = getattr(self.runtime.queue, "ack", None)
                if callable(acknowledge):
                    acknowledge(step.queued_message_id, lease_owner=self.daemon_id)
            if step.status == LoopStepStatus.PROCESSED:
                self._mark_snapshot_checkpoint_terminal(snapshot, status="done")
                self._close_terminal_task_progress(snapshot)
        elif step.status == LoopStepStatus.CANCELLED and step.queued_message_id:
            acknowledge = getattr(self.runtime.queue, "ack", None)
            if callable(acknowledge):
                acknowledge(step.queued_message_id, lease_owner=self.daemon_id)
            self._mark_snapshot_checkpoint_terminal(snapshot, status="cancelled")
            self._close_terminal_task_progress(snapshot)
        elif step.status == LoopStepStatus.FAILED and step.queued_message_id:
            error = str(getattr(step, "error", "") or "capd_processing_failed")
            metadata = getattr(getattr(queued, "message", None), "metadata", {}) or {}
            scheduled_occurrence = str(metadata.get("occurrence_key") or "").strip() if isinstance(metadata, dict) and str(metadata.get("source") or "") == "scheduled_task" else ""
            non_retryable = _scheduled_failure_is_non_retryable(error)
            retry = getattr(self.runtime.queue, "retry", None)
            if callable(retry):
                retry(
                    step.queued_message_id,
                    error=error,
                    next_attempt_at=datetime.now(timezone.utc) + timedelta(seconds=5),
                    lease_owner=self.daemon_id,
                    max_attempts=1 if non_retryable else self.inbound_max_attempts,
                )
            status_for = getattr(self.runtime.queue, "status_for", None)
            terminal = non_retryable or (callable(status_for) and status_for(step.queued_message_id) == "failed")
            if scheduled_occurrence and terminal:
                self.runtime.schedule_store.mark_occurrence_processing_failed(scheduled_occurrence, error=error)
                self._notify_scheduled_task_failure(metadata, error=error)
            self.runtime.core.clear_failure()
        if step.status != LoopStepStatus.BUSY:
            self.kill_switch.clear(str(step.queued_message_id or ""))
            self._set_active_work(None)
            terminal_state = {
                LoopStepStatus.WAITING: "waiting",
                LoopStepStatus.PARKED: "waiting",
                LoopStepStatus.FAILED: "error",
            }.get(step.status, "idle")
            self._set_activity_status(terminal_state)
        return step

    def _project_snapshot_delivery(
        self,
        snapshot: Any,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> Any:
        projected = project_snapshot_to_outbox(
            snapshot=snapshot,
            outbox=self.runtime.outbox,
            identity_resolver=self.runtime.identity_resolver,
            mirror_automation_messages_to_preferred_channel=self.runtime.user_store.mirror_automation_messages_to_preferred_channel(),
            connection=connection,
        )
        if projected is not None:
            self.runtime.conversation_store.record(
                owner_user_id=projected.audience_user_id,
                project_id=projected.project_id,
                role="assistant",
                content=projected.message,
                source=projected.integration_id,
                source_message_id=f"outbound:{projected.outbox_message_id}",
                created_at=projected.created_at,
                connection=connection,
            )
        return projected

    def _mark_projected_occurrence(
        self,
        projected: Any,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if projected is None:
            return
        occurrence_key = str(projected.metadata.get("occurrence_key") or "").strip()
        if occurrence_key:
            arguments = {"response_outbox_id": projected.outbox_message_id}
            if connection is not None:
                arguments["connection"] = connection
            self.runtime.schedule_store.mark_occurrence_response_pending(occurrence_key, **arguments)

    def _deliver_scheduled_reminder(self, occurrence: Any) -> str:
        """Deliver notification-only schedules without starting an LLM/PDCA run."""
        task = occurrence.task
        origin = channel_address_from_metadata({"channel": dict(task.origin_channel)})
        if origin is None:
            resolved = self.runtime.identity_resolver.resolve_outbound_address(
                alphonse_user_id=task.owner_user_id,
            )
            origin = resolved.address if resolved.resolved else None
        if origin is None:
            raise RuntimeError("scheduled_reminder_delivery_unresolved")
        message = str(task.description or task.name or "Reminder").strip()
        outbound = self.runtime.outbox.enqueue(
            address=origin,
            message=f"Reminder: {message}",
            kind="scheduled_reminder",
            audience_user_id=task.owner_user_id,
            project_id=task.project_id,
            metadata={
                "source": "scheduled_reminder",
                "scheduled_task_id": task.scheduled_task_id,
                "occurrence_key": occurrence.occurrence_key,
            },
        )
        self.runtime.conversation_store.record(
            owner_user_id=task.owner_user_id,
            project_id=task.project_id,
            role="assistant",
            content=outbound.message,
            source=origin.integration_id,
            source_message_id=f"outbound:{outbound.outbox_message_id}",
        )
        return outbound.outbox_message_id

    def _notify_scheduled_task_failure(self, metadata: dict[str, Any], *, error: str) -> None:
        """Notify the owner when scheduled work cannot run, without involving the model."""
        task_id = str(metadata.get("scheduled_task_id") or "").strip()
        task = self.runtime.schedule_store.get_task(task_id)
        if task is None:
            return
        origin = channel_address_from_metadata({"channel": metadata.get("channel")})
        if origin is None:
            resolved = self.runtime.identity_resolver.resolve_outbound_address(alphonse_user_id=task.owner_user_id)
            origin = resolved.address if resolved.resolved else None
        if origin is None:
            logger.error("scheduled task failure could not be delivered task_id=%s error=%s", task_id, error)
            return
        message = _scheduled_failure_message(task.name, error)
        outbound = self.runtime.outbox.enqueue(
            address=origin,
            message=message,
            kind="scheduled_task_failed",
            audience_user_id=task.owner_user_id,
            project_id=task.project_id,
            metadata={"source": "scheduled_task_failure", "scheduled_task_id": task_id, "failure_code": _scheduled_failure_code(error)},
        )
        self.runtime.conversation_store.record(
            owner_user_id=task.owner_user_id,
            project_id=task.project_id,
            role="assistant",
            content=message,
            source=origin.integration_id,
            source_message_id=f"outbound:{outbound.outbox_message_id}",
        )

    def _run_retention_if_due(self, *, force: bool = False) -> None:
        now = datetime.now(timezone.utc)
        if not force and self._last_retention_at is not None and now - self._last_retention_at < timedelta(days=1):
            return
        db_path = str(getattr(self.runtime.user_store, "db_path", ":memory:"))
        if db_path != ":memory:":
            deleted = prune_operational_data(db_path, now=now)
            logger.info("operational retention completed deleted=%s", deleted)
        self._last_retention_at = now

    def _mark_snapshot_checkpoint_terminal(self, snapshot: Any, *, status: str) -> None:
        metadata = getattr(snapshot, "metadata", {}) if snapshot is not None else {}
        task_state = metadata.get("task_state") if isinstance(metadata, dict) else None
        task_id = str(task_state.get("task_id") or "").strip() if isinstance(task_state, dict) else ""
        if task_id:
            self.runtime.question_store.mark_task_checkpoint_terminal(task_id, status=status)

    def _close_terminal_task_progress(self, snapshot: Any) -> None:
        """Close Desktop progress cards even when CAPD produces no outbound reply."""
        metadata = getattr(snapshot, "metadata", {}) if snapshot is not None else {}
        task_state = metadata.get("task_state") if isinstance(metadata, dict) else None
        if not isinstance(task_state, dict):
            return
        task_id = str(task_state.get("task_id") or "").strip()
        owner = str(task_state.get("user") or "").strip()
        if not task_id or not owner:
            return
        for key, known in self._desktop_progress_surfaces.items():
            if key[1] == owner and task_id in known:
                self._desktop_progress_closures.setdefault(key, set()).add(task_id)

    def _set_activity_status(self, state: str) -> None:
        with self._active_work_lock:
            self._activity_status = {"state": state, "updated_at": datetime.now(timezone.utc).isoformat()}

    def _set_active_work(self, queued: Any | None) -> None:
        with self._active_work_lock:
            if queued is None:
                self._active_work = {}
                return
            message = getattr(queued, "message", None)
            self._active_work = {
                "message_id": str(getattr(queued, "message_id", "") or ""),
                "user": str(getattr(message, "user", "") or ""),
                "project_id": str(getattr(message, "project_id", "") or ""),
                "correlation_id": str(getattr(message, "correlation_id", "") or ""),
                "prompt": str(getattr(message, "prompt", "") or ""),
                "routing_disposition": str((getattr(message, "metadata", {}) or {}).get("routing_disposition") or "pdca_task"),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.wait(0.5):
                pass
        finally:
            self.stop()

    def _emit_scheduled_task_card(self, snapshot: Any) -> None:
        metadata = getattr(snapshot, "metadata", {}) or {}
        task_state = metadata.get("task_state") if isinstance(metadata, dict) else None
        result = _scheduled_task_result(task_state)
        if result is None:
            return
        task_metadata = task_state.get("metadata") if isinstance(task_state.get("metadata"), dict) else {}
        channel = task_metadata.get("channel") if isinstance(task_metadata.get("channel"), dict) else {}
        if str(channel.get("integration_id") or "") != "desktop":
            return
        record = self.runtime.schedule_store.get_task(str(result.get("scheduled_task_id") or ""))
        if record is None:
            return
        project_id = str(task_state.get("project_id") or "").strip()
        project = self.runtime.project_store.get_project(project_id, requester_user_id=str(task_state.get("user") or ""), requester_is_admin=True) if project_id else None
        self.runtime.ui_events.append(
            CoreUiEvent(
                event_type="scheduled_task_created",
                payload={
                    "task_state": task_state,
                    "scheduled_task": {**record.to_dict(), "schedule_summary": schedule_summary(record.schedule)},
                    "project_name": project.name if project is not None else "",
                },
            )
        )

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            try:
                step = self.run_once()
                self._last_processor_error = ""
            except Exception as exc:
                self._last_processor_error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                self._stop.wait(max(0.1, self.poll_interval_sec))
                continue
            if step.status in {LoopStepStatus.EMPTY, LoopStepStatus.BUSY}:
                self._stop.wait(max(0.01, self.poll_interval_sec))

    def _acquire_single_instance_lock(self) -> None:
        if self._lock_file is not None:
            return
        self._lock_file = _claim_single_instance_lock(self.daemon_id)

    def _release_single_instance_lock(self) -> None:
        with self._lifecycle_lock:
            lock_file = self._lock_file
            self._lock_file = None
        if lock_file is None:
            return
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
        finally:
            pass

    def _on_outbox_delivered(self, outbound: Any) -> None:
        message_id = str(getattr(outbound, "outbox_message_id", "") or "")
        if message_id:
            delivered = self.runtime.outbox.get(message_id)
            self.runtime.communication_router.threads.mark_delivered(
                message_id,
                str(getattr(delivered, "provider_message_id", "") or ""),
            )
        metadata = getattr(outbound, "metadata", {}) if outbound is not None else {}
        occurrence_key = str(metadata.get("occurrence_key") or "").strip() if isinstance(metadata, dict) else ""
        if occurrence_key:
            self.runtime.schedule_store.mark_occurrence_delivered(
                occurrence_key,
                response_outbox_id=str(getattr(outbound, "outbox_message_id", "") or ""),
            )

    def _on_outbox_failed(self, outbound: Any, error: str) -> None:
        message_id = str(getattr(outbound, "outbox_message_id", "") or "")
        if message_id:
            thread = self.runtime.communication_router.threads.mark_failed(message_id)
            if thread is not None:
                self.runtime.outbox.enqueue(
                    address=thread.origin,
                    message="I could not deliver your message. Please try again later.",
                    kind="communication_delivery_failed",
                    audience_user_id=thread.sender_user_id,
                    metadata={"communication_thread_id": thread.thread_id},
                )
        metadata = getattr(outbound, "metadata", {}) if outbound is not None else {}
        if isinstance(metadata, dict) and metadata.get("automation_preferred_channel_copy"):
            logger.warning(
                "automation preferred-channel copy delivery failed correlation_id=%s error=%s",
                str(getattr(outbound, "correlation_id", "") or ""),
                error,
            )
            return
        occurrence_key = str(metadata.get("occurrence_key") or "").strip() if isinstance(metadata, dict) else ""
        if occurrence_key:
            self.runtime.schedule_store.mark_occurrence_failed(occurrence_key, error=error)


def _without_sequence(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "sequence"}


def _conversation_message_id(source_message_id: str, fallback_event_id: str) -> str:
    value = str(source_message_id or "").strip()
    for prefix in ("inbound:", "outbound:"):
        if value.startswith(prefix):
            return value.removeprefix(prefix)
    return value or str(fallback_event_id or "").strip()


def _ui_event_user(event: CoreUiEvent) -> str:
    payload = dict(event.payload or {})
    question = payload.get("question")
    if isinstance(question, dict):
        return str(question.get("respondent_user_id") or "").strip()
    task = payload.get("task") or payload.get("task_state")
    if isinstance(task, dict):
        return str(task.get("user") or "").strip()
    return str(payload.get("user") or "").strip()


def _a2ui_custom(envelope: dict[str, Any]) -> dict[str, Any]:
    return {"type": "CUSTOM", "name": "a2ui.envelope", "value": envelope}


def _event_name(item: dict[str, Any]) -> str:
    event = item.get("event") if isinstance(item.get("event"), dict) else {}
    return str(event.get("name") or "")


def _local_datetime_to_utc(value: datetime, *, timezone_name: str) -> str:
    """Convert an unambiguous local wall time to a canonical UTC instant."""
    zone = ZoneInfo(str(timezone_name or "UTC"))
    candidates = {
        candidate.astimezone(timezone.utc)
        for fold in (0, 1)
        if (candidate := value.replace(tzinfo=zone, fold=fold)).astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) == value
    }
    if len(candidates) != 1:
        raise ValueError("local_datetime_ambiguous_or_invalid")
    return next(iter(candidates)).isoformat().replace("+00:00", "Z")


def _scheduled_task_id_from_surface(surface_id: str) -> str:
    value = str(surface_id or "").strip()
    return value.removeprefix("scheduled-task:") if value.startswith("scheduled-task:") else ""


def _scheduled_failure_code(error: str) -> str:
    value = str(error or "").strip()
    return value.split(":", 1)[0] or "scheduled_task_processing_failed"


def _scheduled_failure_is_non_retryable(error: str) -> bool:
    return _scheduled_failure_code(error) in {
        "openai_codex_auth_required",
        "openai_codex_cli_missing",
        "openai_codex_cli_upgrade_required",
    }


def _scheduled_failure_message(task_name: str, error: str) -> str:
    code = _scheduled_failure_code(error)
    if code == "openai_codex_auth_required":
        detail = "Codex needs to be signed in again before I can run it."
    elif code == "openai_codex_cli_missing":
        detail = "The Codex command-line tool is unavailable on this machine."
    elif code == "openai_codex_cli_upgrade_required":
        detail = "Codex needs to be updated before I can run it."
    else:
        detail = "I could not complete it after retrying."
    return f"Scheduled task failed: {task_name}. {detail}"


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _event_payload(execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_key": str(execution.get("event_key") or ""),
        "worker_id": str(execution.get("worker_id") or ""),
        "event_id": str(execution.get("source_event_id") or ""),
        "event_type": str(execution.get("event_type") or ""),
        "event_version": str(execution.get("event_version") or ""),
        "occurred_at": str(execution.get("occurred_at") or ""),
        "payload": _json_object(execution.get("payload_json")),
    }


def _scheduled_task_result(task_state: Any) -> dict[str, Any] | None:
    if not isinstance(task_state, dict):
        return None
    calls = task_state.get("plan_json")
    try:
        import json
        calls = json.loads(calls) if isinstance(calls, str) else calls
    except (TypeError, ValueError):
        return None
    if not isinstance(calls, list):
        return None
    for call in reversed(calls):
        if not isinstance(call, dict) or str(call.get("tool_id") or "") != "native.scheduled_task":
            continue
        execution = call.get("execution") if isinstance(call.get("execution"), dict) else {}
        result = execution.get("result") if isinstance(execution.get("result"), dict) else {}
        if str(execution.get("status") or "") == "success" and str(result.get("scheduled_task_id") or ""):
            return result
    return None


def _comma_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _positive_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 1.0
    return parsed if parsed > 0 else 1.0


def _question_answer_text(payload: dict[str, Any] | None) -> str:
    value = dict(payload or {})
    if "text" in value:
        return str(value["text"] or "")
    if "choice_id" in value:
        return str(value["choice_id"] or "")
    if "choice_ids" in value:
        return ", ".join(str(item) for item in value["choice_ids"] or [])
    if "datetime" in value:
        return str(value["datetime"] or "")
    if "answer" in value:
        return "yes" if bool(value["answer"]) else "no"
    return "answer"


def main() -> None:
    daemon_id = f"daemon-{uuid4().hex[:12]}"
    startup_lock = _claim_single_instance_lock(daemon_id)
    try:
        migration = migrate_legacy_databases()
        logger.info(
            "storage migration status=%s target=%s sources=%s",
            migration.get("status"),
            migration.get("target"),
            len(migration.get("sources") or []),
        )
        user_store = V2UserStore.default()
        asset_store = SQLiteAssetStore.default(users_root=user_store.users_root)
        asset_store.migrate_to_user_directories()
        daemon = V2Daemon(
            build_runtime_host(
                user=user_store.admin_user().user_id if user_store.admin_user() else "",
                user_store=user_store,
                messages=SQLiteMessageQueue.default(lease_owner=daemon_id),
                question_store=SQLiteQuestionStore.default(),
                project_store=ProjectStore.default(),
                schedule_store=ScheduledTaskStore.default(),
                web_tools_settings_store=SQLiteWebToolsSettingsStore.default(),
                code_mode_settings_store=SQLiteCodeModeSettingsStore.default(),
                media_tools_settings_store=SQLiteMediaToolsSettingsStore.default(),
                asset_store=asset_store,
                artifact_store=SQLiteArtifactStore.default(),
                conversation_store=SQLiteConversationStore.default(),
                memory_settings_store=SQLiteMemorySettingsStore.default(),
                outbox=SQLiteOutboundStore.default(),
                integration_store=SQLiteIntegrationStore.default(),
                inference_settings_store=SQLiteInferenceSettingsStore.default(),
                agent_config_store=AgentConfigStore.default(),
                project_session_store=SQLiteProjectSessionStore.default(),
                communication_thread_store=SQLiteCommunicationThreadStore.default(),
            ),
            daemon_id=daemon_id,
        )
        daemon._lock_file = startup_lock
        startup_lock = None
    finally:
        if startup_lock is not None:
            fcntl.flock(startup_lock.fileno(), fcntl.LOCK_UN)
            startup_lock.close()
    previous = {}

    def _request_stop(signum: int, frame: Any) -> None:
        _ = signum, frame
        daemon.stop()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, _request_stop)
    try:
        daemon.run_forever()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
