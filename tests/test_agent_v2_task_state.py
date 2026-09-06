from __future__ import annotations

import json
from datetime import datetime

import pytest

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.intelligence import TaskState
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue


def test_from_message_maps_canonical_core_message_fields() -> None:
    message = CoreMessage(
        timestamp=datetime.now().astimezone(),
        prompt="Build the file",
        user="gaby",
        project_id="home",
        tag="writing",
    )

    state = TaskState.from_message(message, message_id="msg-1")

    assert state.message_id == "msg-1"
    assert state.goal == "Build the file"
    assert state.user == "gaby"
    assert state.project_id == "home"
    assert state.tag == "writing"
    assert state.recent_conversation_md == '- gaby: "Build the file"'


def test_from_message_carries_message_metadata() -> None:
    message = CoreMessage(
        timestamp=datetime.now().astimezone(),
        prompt="/project new",
        user="alex",
        metadata={"is_command": True, "command": "project", "command_args": "new"},
    )

    state = TaskState.from_message(message)

    assert {key: state.metadata[key] for key in ("is_command", "command", "command_args")} == {
        "is_command": True,
        "command": "project",
        "command_args": "new",
    }


def test_resumed_task_keeps_new_owner_attachments():
    queue = InMemoryMessageQueue()
    old = TaskState(user="alex", project_id="alpha", metadata={"asset_ids": ["first"]})
    queued = CommunicationChannel(queue).queue_message(prompt="Here is the receipt", user="alex", project_id="alpha", metadata={"task_state": old.to_dict(), "attachments": [{"asset_id": "receipt"}]})
    state = TaskState.from_queued_message(queued)
    assert state.metadata["asset_ids"] == ["first", "receipt"]
    state.merge_attachments({"attachments": [{"asset_id": "receipt"}]})
    assert state.metadata["attachments"] == [{"asset_id": "receipt"}]


def test_append_conversation_message_uses_speaker_and_quoted_prompt() -> None:
    state = TaskState()

    state.append_conversation_message("Alex", "Hi Alphonse!")
    state.append_conversation_message("Alphonse [agent]", "Hello, sir.")

    assert state.recent_conversation_md == '- Alex: "Hi Alphonse!"\n- Alphonse [agent]: "Hello, sir."'


def test_from_queued_message_preserves_queue_and_message_metadata() -> None:
    queue = InMemoryMessageQueue()
    queued = CommunicationChannel(queue).queue_message(
        prompt="/project new",
        user="alex",
        project_id="alpha",
        tag="work",
    )

    state = TaskState.from_queued_message(queued)

    assert state.message_id == queued.message_id
    assert state.goal == "/project new"
    assert state.user == "alex"
    assert state.project_id == "alpha"
    assert state.tag == "work"
    assert {key: state.metadata[key] for key in ("is_command", "command", "command_args")} == {
        "is_command": True,
        "command": "project",
        "command_args": "new",
    }
    assert state.metadata["alphonse_user_id"] == "alex"
    assert state.metadata["channel"]["integration_id"] == "tui"
    assert state.metadata["channel"]["channel_target"] == "alex"


def test_markdown_defaults_are_none_bullets() -> None:
    state = TaskState()

    assert state.facts_md == "- (none)"
    assert state.recent_conversation_md == "- (none)"
    assert state.plan_json == "- (none)"
    assert state.acceptance_criteria_md == "- (none)"
    assert state.memory_facts_md == "- (none)"
    assert state.updates_md == "- (none)"


def test_append_helpers_use_bullet_markdown() -> None:
    state = TaskState()

    state.append_fact("fact one")
    state.append_fact("- fact two")
    state.append_plan_call(
        {
            "id": "plan-call-1",
            "tool_id": "tool-1",
            "tool_name": "write_file",
            "arguments": {"path": "a.txt"},
            "internal_state": "Writing the requested file.",
        }
    )
    state.append_acceptance_criterion("criterion one")
    state.append_memory_fact("memory one")
    state.append_recent_conversation_line("conversation one")
    state.append_update("update one")

    assert state.facts_md == "- fact one\n- fact two"
    assert json.loads(state.plan_json) == [
        {
            "id": "plan-call-1",
            "tool_id": "tool-1",
            "tool_name": "write_file",
            "arguments": {"path": "a.txt"},
            "internal_state": "Writing the requested file.",
        }
    ]
    assert state.acceptance_criteria_md == "- criterion one"
    assert state.memory_facts_md == "- memory one"
    assert state.recent_conversation_md == "- conversation one"
    assert state.updates_md == "- update one"


def test_to_dict_from_dict_round_trip() -> None:
    state = TaskState(
        task_id="task-1",
        message_id="msg-1",
        user="alex",
        project_id="alpha",
        tag="core",
        correlation_id="corr-1",
        goal="Goal",
        status="done",
        outcome={"ok": True},
        check_verdict="wip",
        check_reason="Still working",
        check_confidence=0.4,
        check_evidence_refs=["ref-1"],
        check_new_message_count=2,
        pdca_cycle_count=3,
        metadata={"is_command": False},
    )
    state.append_update("updated")

    restored = TaskState.from_dict(state.to_dict())

    assert restored.to_dict() == state.to_dict()


def test_from_dict_reads_legacy_plan_md_fallback() -> None:
    state = TaskState.from_dict({"goal": "Goal", "plan_md": "- legacy plan"})

    assert state.plan_json == "- legacy plan"
    assert "plan_md" not in state.to_dict()
    assert state.to_dict()["plan_json"] == "- legacy plan"


def test_set_check_result_validates_verdict_and_clamps_confidence() -> None:
    state = TaskState()

    state.set_check_result(
        verdict="MISSION_SUCCESS",
        reason="done",
        confidence=2.5,
        evidence_refs=[" ref-1 ", ""],
        new_message_count=-1,
    )

    assert state.check_verdict == "mission_success"
    assert state.check_reason == "done"
    assert state.check_confidence == 1.0
    assert state.check_evidence_refs == ["ref-1"]
    assert state.check_new_message_count == 0

    with pytest.raises(ValueError, match="invalid_check_verdict"):
        state.set_check_result(verdict="unknown")


def test_replan_clears_goal_acceptance_criteria_status_and_outcome() -> None:
    state = TaskState(
        goal="old",
        acceptance_criteria_md="- pass",
        status="failed",
        outcome={"error": "no"},
    )

    state.replan()

    assert state.goal == ""
    assert state.acceptance_criteria_md == "- (none)"
    assert state.status == "running"
    assert state.outcome is None


def test_to_markdown_prompt_includes_expected_sections_without_mutating() -> None:
    state = TaskState(task_id="task-1", user="alex", goal="Create a project")
    state.append_plan_call(
        {
            "id": "plan-call-1",
            "tool_id": "tool-1",
            "tool_name": "write_file",
            "arguments": {"path": "a.txt"},
            "internal_state": "Writing the project file.",
        }
    )
    before = state.to_dict()

    rendered = state.to_markdown_prompt()

    assert "# Task Metadata" in rendered
    assert "# Goal" in rendered
    assert "Create a project" in rendered
    assert "# Recent Conversation" in rendered
    assert "# Facts" in rendered
    assert "# Plan JSON" in rendered
    assert "plan-call-1" in rendered
    assert "# Acceptance Criteria" in rendered
    assert "# Updates" in rendered
    assert "# Memory Facts" in rendered
    assert "# Tool Call History" not in rendered
    assert "# Check Result" in rendered
    assert "# Outcome" in rendered
    assert state.to_dict() == before


def test_get_next_planned_call_returns_latest_unexecuted_call() -> None:
    first = {
        "id": "plan-call-1",
        "tool_id": "tool-1",
        "tool_name": "write_file",
        "arguments": {},
        "internal_state": "First call.",
    }
    second = {
        "id": "plan-call-2",
        "tool_id": "tool-2",
        "tool_name": "read_file",
        "arguments": {},
        "internal_state": "Second call.",
    }
    state = TaskState()
    state.append_plan_call(first)
    state.append_plan_call(second)

    assert state.get_next_planned_call() == second


def test_get_next_planned_call_skips_executed_calls() -> None:
    first = {
        "id": "plan-call-1",
        "tool_id": "tool-1",
        "tool_name": "write_file",
        "arguments": {},
        "internal_state": "First call.",
    }
    second = {
        "id": "plan-call-2",
        "tool_id": "tool-2",
        "tool_name": "read_file",
        "arguments": {},
        "internal_state": "Second call.",
    }
    state = TaskState()
    state.append_plan_call(first)
    state.append_plan_call(second)
    state.record_plan_call_success("plan-call-2", {"ok": True})

    assert state.get_next_planned_call() == first


def test_record_plan_call_success_updates_matching_plan_row() -> None:
    state = TaskState()
    state.append_plan_call(
        {
            "id": "plan-call-1",
            "tool_id": "tool-1",
            "tool_name": "write_file",
            "arguments": {},
            "internal_state": "Writing.",
        }
    )

    state.record_plan_call_success("plan-call-1", {"ok": True})

    execution = json.loads(state.plan_json)[0]["execution"]
    assert execution["status"] == "success"
    assert execution["result"] == {"ok": True}
    assert execution["exception"] == ""
    assert execution["started_at"]
    assert execution["finished_at"]


def test_record_plan_call_exception_updates_matching_plan_row() -> None:
    state = TaskState()
    state.append_plan_call(
        {
            "id": "plan-call-1",
            "tool_id": "tool-1",
            "tool_name": "write_file",
            "arguments": {},
            "internal_state": "Writing.",
        }
    )

    state.record_plan_call_exception("plan-call-1", RuntimeError("boom"))

    execution = json.loads(state.plan_json)[0]["execution"]
    assert execution["status"] == "exception"
    assert execution["result"] is None
    assert execution["exception"] == "RuntimeError: boom"


def test_get_next_planned_call_handles_default_or_invalid_plan_json() -> None:
    assert TaskState().get_next_planned_call() is None
    assert TaskState(plan_json="not json").get_next_planned_call() is None


def test_get_latest_executed_plan_call_returns_most_recent_executed_call() -> None:
    first = {
        "id": "plan-call-1",
        "tool_id": "tool-1",
        "tool_name": "write_file",
        "arguments": {},
        "internal_state": "First call.",
    }
    second = {
        "id": "plan-call-2",
        "tool_id": "tool-2",
        "tool_name": "read_file",
        "arguments": {},
        "internal_state": "Second call.",
    }
    state = TaskState()
    state.append_plan_call(first)
    state.append_plan_call(second)
    state.record_plan_call_success("plan-call-1", {"ok": "first"})
    state.record_plan_call_success("plan-call-2", {"ok": "second"})

    latest = state.get_latest_executed_plan_call()

    assert latest is not None
    assert latest["id"] == "plan-call-2"
    assert latest["execution"]["result"] == {"ok": "second"}


def test_get_latest_executed_plan_call_ignores_unexecuted_calls() -> None:
    state = TaskState()
    state.append_plan_call(
        {
            "id": "plan-call-1",
            "tool_id": "tool-1",
            "tool_name": "write_file",
            "arguments": {},
            "internal_state": "First call.",
        }
    )
    state.append_plan_call(
        {
            "id": "plan-call-2",
            "tool_id": "tool-2",
            "tool_name": "read_file",
            "arguments": {},
            "internal_state": "Second call.",
        }
    )
    state.record_plan_call_success("plan-call-1", {"ok": True})

    latest = state.get_latest_executed_plan_call()

    assert latest is not None
    assert latest["id"] == "plan-call-1"


def test_get_latest_executed_plan_call_handles_default_invalid_or_no_execution_plan_json() -> None:
    assert TaskState().get_latest_executed_plan_call() is None
    assert TaskState(plan_json="not json").get_latest_executed_plan_call() is None
    state = TaskState()
    state.append_plan_call(
        {
            "id": "plan-call-1",
            "tool_id": "tool-1",
            "tool_name": "write_file",
            "arguments": {},
            "internal_state": "First call.",
        }
    )
    assert state.get_latest_executed_plan_call() is None


def test_acceptance_criteria_all_complete_requires_real_completed_checkbox_criteria() -> None:
    assert TaskState(acceptance_criteria_md="1.- [x] File exists").acceptance_criteria_all_complete()
    assert TaskState(acceptance_criteria_md="1.- [X] File exists\nnotes without checkbox").acceptance_criteria_all_complete()
    assert not TaskState(acceptance_criteria_md="1.- [ ] File exists").acceptance_criteria_all_complete()
    assert not TaskState(acceptance_criteria_md="- (none)").acceptance_criteria_all_complete()
    assert not TaskState(acceptance_criteria_md="plain text only").acceptance_criteria_all_complete()


def test_count_plan_call_exceptions_counts_exception_executions() -> None:
    state = TaskState()
    for index in range(3):
        state.append_plan_call(
            {
                "id": f"plan-call-{index}",
                "tool_id": "tool-1",
                "tool_name": "write_file",
                "arguments": {},
                "internal_state": "Writing.",
            }
        )
    state.record_plan_call_exception("plan-call-0", RuntimeError("first"))
    state.record_plan_call_success("plan-call-1", {"ok": True})
    state.record_plan_call_exception("plan-call-2", RuntimeError("second"))

    assert state.count_plan_call_exceptions() == 2
