from __future__ import annotations

from importlib import import_module

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import StubInferenceProvider
from alphonse.agent_v2.core.intelligence.pdca.nodes import check_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue

check_node_module = import_module("alphonse.agent_v2.core.intelligence.pdca.nodes.check_node")


def test_check_node_marks_new_task_when_acceptance_criteria_are_empty() -> None:
    task = TaskState(goal="Write a file")

    result = check_node(task)

    assert result is task
    assert task.check_verdict == "new"
    assert task.check_new_message_count == 0


def test_check_node_marks_existing_task_without_steering_as_wip() -> None:
    task = TaskState(goal="Continue task", acceptance_criteria_md="- File exists")

    check_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue()))

    assert task.check_verdict == "wip"
    assert task.check_new_message_count == 0
    assert "criteria_review_prompt" not in task.metadata


def test_check_node_consumes_same_user_same_project_steering() -> None:
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)
    channel.queue_message(prompt="Add tests too", user="alex", project_id="alpha", metadata={"routing_disposition": "steering"})
    channel.queue_message(prompt="Different project", user="alex", project_id="beta")
    task = TaskState(
        goal="Continue task",
        user="alex",
        project_id="alpha",
        acceptance_criteria_md="- Feature works",
    )

    check_node(task, context=CoreLoopContext(messages=queue))

    assert task.check_verdict == "steer"
    assert task.check_new_message_count == 1
    assert '- alex: "Add tests too"' in task.recent_conversation_md
    assert "Different project" not in task.recent_conversation_md
    assert queue.size() == 1


def test_check_node_leaves_scheduled_occurrences_for_independent_processing() -> None:
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)
    channel.queue_message(
        prompt="Time to drink water.",
        user="alex",
        project_id="alpha",
        metadata={"source": "scheduled_task", "scheduled_task_id": "scheduled-task-1"},
    )
    task = TaskState(
        goal="Continue task",
        user="alex",
        project_id="alpha",
        acceptance_criteria_md="- Feature works",
    )

    check_node(task, context=CoreLoopContext(messages=queue))

    assert task.check_new_message_count == 0
    assert queue.size() == 1


def test_check_node_steer_reviews_acceptance_criteria_before_returning(monkeypatch) -> None:
    queue = InMemoryMessageQueue()
    CommunicationChannel(queue).queue_message(prompt="The file exists now", user="alex", project_id="alpha", metadata={"routing_disposition": "steering"})
    task = TaskState(
        goal="Create a file",
        user="alex",
        project_id="alpha",
        acceptance_criteria_md="1.- [ ] File exists",
    )
    monkeypatch.setattr(check_node_module, "_call_criteria_review_llm", lambda prompt: "1.- [x] File exists")

    events = []
    check_node(task, context=CoreLoopContext(messages=queue, activity_sink=events.append))

    assert task.check_verdict == "steer"
    assert task.acceptance_criteria_md == "1.- [x] File exists"
    assert task.metadata["criteria_review_updated"] is True
    assert "criteria_review_prompt" in task.metadata
    assert events[-1].label == "criteria refreshed"
    assert events[-1].progress["acceptance_criteria"] == "1.- [x] File exists"


def test_check_node_consumes_same_correlation_id_from_other_user() -> None:
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)
    channel.queue_message(prompt="No coffee today", user="Gaby", project_id="home", correlation_id="coffee-1", metadata={"routing_disposition": "correlated_response"})
    task = TaskState(
        goal="Ask Gaby about coffee",
        user="Alex",
        project_id="home",
        correlation_id="coffee-1",
        acceptance_criteria_md="- Answer Alex",
    )

    check_node(task, context=CoreLoopContext(messages=queue))

    assert task.check_verdict == "steer"


def test_check_node_wip_with_latest_execution_renders_criteria_review_prompt() -> None:
    task = _task_with_successful_tool_execution()

    check_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue()))

    assert task.check_verdict == "wip"
    assert task.metadata["criteria_review_llm_stubbed"] is True
    assert task.metadata["criteria_review_updated"] is False
    assert "criteria_review_prompt" in task.metadata
    assert "plan-call-1" in task.metadata["criteria_review_prompt"]
    assert "created" in task.metadata["criteria_review_prompt"]
    assert "Check prepared acceptance criteria review" in task.updates_md


def test_check_node_stubbed_criteria_review_leaves_criteria_unchanged() -> None:
    task = _task_with_successful_tool_execution()
    original = task.acceptance_criteria_md

    check_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue()))

    assert task.acceptance_criteria_md == original
    assert task.check_verdict == "wip"


def test_check_node_criteria_review_can_mark_criteria_complete(monkeypatch) -> None:
    task = _task_with_successful_tool_execution()
    revised = "1.- [x] File exists"
    monkeypatch.setattr(check_node_module, "_call_criteria_review_llm", lambda prompt: revised)

    check_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue()))

    assert task.acceptance_criteria_md == revised
    assert task.metadata["criteria_review_updated"] is True
    assert task.check_verdict == "wip"
    assert "Check updated acceptance criteria" in task.updates_md


def test_check_node_does_not_set_terminal_verdict_when_all_criteria_complete(monkeypatch) -> None:
    task = _task_with_successful_tool_execution()
    monkeypatch.setattr(check_node_module, "_call_criteria_review_llm", lambda prompt: "1.- [x] File exists")

    check_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue()))

    assert task.acceptance_criteria_md == "1.- [x] File exists"
    assert task.check_verdict == "wip"
    assert task.check_verdict not in {"mission_success", "mission_failed"}


def test_check_node_uses_inference_without_tools_for_criteria_review() -> None:
    provider = StubInferenceProvider(markdown_by_purpose={InferencePurpose.CRITERIA_REVIEW: "1.- [x] File exists"})
    router = InferenceRouter(
        provider=provider,
        default_profile=ModelProfile(provider="openai", model="gpt", profile_id="default"),
    )
    task = _task_with_successful_tool_execution()

    check_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), inference=router))

    assert task.acceptance_criteria_md == "1.- [x] File exists"
    assert task.metadata["criteria_review_llm_stubbed"] is False
    assert task.metadata["criteria_review_model_profile"] == "default"
    assert provider.requests[0].purpose == InferencePurpose.CRITERIA_REVIEW
    assert provider.requests[0].tools == ()


def _task_with_successful_tool_execution() -> TaskState:
    task = TaskState(
        goal="Create a file",
        user="alex",
        project_id="alpha",
        acceptance_criteria_md="1.- [ ] File exists",
    )
    task.append_plan_call(
        {
            "id": "plan-call-1",
            "tool_id": "tool-1",
            "tool_name": "write_file",
            "arguments": {"path": "a.txt"},
            "internal_state": "Writing the requested file.",
        }
    )
    task.record_plan_call_success("plan-call-1", {"path": "a.txt", "status": "created"})
    return task
    assert task.check_new_message_count == 1
    assert '- Gaby: "No coffee today"' in task.recent_conversation_md
    assert queue.size() == 0


def test_check_node_keeps_new_verdict_after_consuming_steering() -> None:
    queue = InMemoryMessageQueue()
    CommunicationChannel(queue).queue_message(prompt="Add this detail", user="alex", project_id="alpha", metadata={"routing_disposition": "steering"})
    task = TaskState(goal="New task", user="alex", project_id="alpha")

    check_node(task, context=CoreLoopContext(messages=queue))

    assert task.check_verdict == "new"
    assert task.check_new_message_count == 1
    assert '- alex: "Add this detail"' in task.recent_conversation_md
    assert queue.size() == 0


def test_steering_image_is_preserved_for_ocr_and_checkpoint_resume():
    queue = InMemoryMessageQueue()
    attachment = {"asset_id": "receipt-1", "filename": "receipt.jpg", "mime_type": "image/jpeg"}
    CommunicationChannel(queue).queue_message(prompt="Aquí está la cuenta", user="alex", project_id="alpha", metadata={"routing_disposition": "steering", "attachments": [attachment], "asset_ids": ["receipt-1"]})
    task = TaskState(goal="Split the receipt", user="alex", project_id="alpha")
    check_node(task, context=CoreLoopContext(messages=queue))
    restored = TaskState.from_dict(task.to_checkpoint_dict())
    assert restored.metadata["asset_ids"] == ["receipt-1"]
    assert restored.metadata["attachments"] == [attachment]
    assert "Asset ID: receipt-1" in restored.to_markdown_prompt()


def test_check_node_does_not_convert_consumed_messages_into_task_states(monkeypatch) -> None:
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)
    channel.queue_message(prompt="Steering", user="alex", project_id="alpha", metadata={"routing_disposition": "steering"})
    task = TaskState(
        goal="Existing task",
        user="alex",
        project_id="alpha",
        acceptance_criteria_md="- Done",
    )

    def fail_from_queued_message(*args: object, **kwargs: object) -> TaskState:
        _ = args
        _ = kwargs
        raise AssertionError("steering messages should not become TaskState objects")

    monkeypatch.setattr(TaskState, "from_queued_message", fail_from_queued_message)

    check_node(task, context=CoreLoopContext(messages=queue))

    assert task.check_verdict == "steer"
