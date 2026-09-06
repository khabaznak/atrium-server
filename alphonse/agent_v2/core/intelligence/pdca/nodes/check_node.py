"""Check node for the v2 PDCA graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import select_autoescape

from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRequest
from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.queue import MessageSelector

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext

_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"


def check_node(task: TaskState, context: CoreLoopContext | None = None) -> TaskState:
    """Classify the task and fold related steering messages into it."""
    if context is not None:
        context.emit_activity(
            phase=ImprovementPhase.CHECK,
            label="deliberating",
            message="Reviewing the task and queued steering messages.",
        )
    is_new_task = _markdown_is_empty(task.acceptance_criteria_md)
    steering_count = _consume_steering_messages(task, context)

    if is_new_task:
        verdict = "new"
        reason = "No acceptance criteria were present; treating this as a new task."
    elif steering_count > 0:
        verdict = "steer"
        reason = _review_wip_acceptance_criteria(task, context, require_latest_call=False)
    else:
        verdict = "wip"
        reason = _review_wip_acceptance_criteria(task, context)

    task.set_check_result(
        verdict=verdict,
        reason=reason,
        confidence=1.0,
        new_message_count=steering_count,
    )
    if context is not None:
        context.emit_activity(
            phase=ImprovementPhase.CHECK,
            label="criteria refreshed",
            message="Acceptance criteria are up to date after review.",
            progress={"acceptance_criteria": "" if task.acceptance_criteria_md == "- (none)" else task.acceptance_criteria_md},
        )
    return task


def _review_wip_acceptance_criteria(
    task: TaskState,
    context: CoreLoopContext | None = None,
    *,
    require_latest_call: bool = True,
) -> str:
    latest_call = task.get_latest_executed_plan_call()
    if latest_call is None and require_latest_call:
        return "Acceptance criteria exist, but no steering messages or executed tool results were available yet."

    prompt = _render_criteria_review_prompt(
        task,
        latest_call or {},
        user_context_md=_user_context_md(task, context),
        project_context_md=_project_context_md(task, context),
        philosophy_md=_agent_prompt_md(context, "Philosophy.md"),
        global_context_md=_agent_prompt_md(context, "GlobalContext.md"),
    )
    revised_criteria = _call_criteria_review_inference(prompt, task, context)
    task.metadata["criteria_review_prompt"] = prompt
    task.metadata["criteria_review_llm_stubbed"] = context is None or context.inference is None
    if revised_criteria:
        task.acceptance_criteria_md = str(revised_criteria).strip()
        task.metadata["criteria_review_updated"] = True
        task.append_update("Check updated acceptance criteria from latest executed tool call.")
        return "Acceptance criteria were reviewed against the latest executed tool call."

    task.metadata["criteria_review_updated"] = False
    task.append_update("Check prepared acceptance criteria review; LLM execution is stubbed.")
    return "Acceptance criteria need review against the latest executed tool call."


def _render_criteria_review_prompt(
    task: TaskState,
    latest_call: dict[str, object],
    *,
    user_context_md: str = "",
    project_context_md: str = "",
    philosophy_md: str = "",
    global_context_md: str = "",
) -> str:
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(default_for_string=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("criteria_review_prompt.j2")
    return template.render(
        acceptance_criteria_md=task.acceptance_criteria_md,
        latest_executed_call_json=json.dumps(latest_call, indent=2, sort_keys=True),
        user_context_md=user_context_md,
        project_context_md=project_context_md,
        philosophy_md=philosophy_md,
        global_context_md=global_context_md,
        task_state_md=task.to_markdown_prompt(),
    ).strip()


def _project_context_md(task: TaskState, context: CoreLoopContext | None) -> str:
    if context is None or context.project_store is None or not str(task.project_id or "").strip():
        return ""
    render = getattr(context.project_store, "render_project_context", None)
    if not callable(render):
        return ""
    return str(render(task.project_id, requester_user_id=task.user) or "").strip()


def _user_context_md(task: TaskState, context: CoreLoopContext | None) -> str:
    if context is None or not callable(context.user_context_provider):
        return ""
    try:
        return str(context.user_context_provider(task.user) or "").strip()
    except (OSError, KeyError):
        return ""


def _agent_prompt_md(context: CoreLoopContext | None, name: str) -> str:
    if context is None or context.prompts is None:
        return ""
    load = getattr(context.prompts, "load", None)
    if not callable(load):
        return ""
    return str(load(name).content or "").strip()


def _call_criteria_review_llm(prompt: str) -> str | None:
    """Stub for the future WIP acceptance criteria review LLM call."""
    _ = prompt
    return None


def _call_criteria_review_inference(
    prompt: str,
    task: TaskState,
    context: CoreLoopContext | None = None,
) -> str | None:
    if context is not None and context.inference is not None:
        result = context.inference.generate_markdown(
            InferenceRequest(
                prompt=prompt,
                purpose=InferencePurpose.CRITERIA_REVIEW,
                project_id=task.project_id,
                user=task.user,
                task_id=task.task_id,
            )
        )
        if result.model_profile is not None:
            task.metadata["criteria_review_model_profile"] = result.model_profile.profile_id
        return str(result.content or "").strip() or None
    return _call_criteria_review_llm(prompt)


def _consume_steering_messages(task: TaskState, context: CoreLoopContext | None) -> int:
    if context is None:
        return 0

    consumed = 0
    consumed += _consume_matching(
        task,
        context,
        MessageSelector(user=task.user, project_id=task.project_id),
    )
    if task.correlation_id:
        consumed += _consume_matching(
            task,
            context,
            MessageSelector(correlation_id=task.correlation_id),
        )
    return consumed


def _consume_matching(task: TaskState, context: CoreLoopContext, selector: MessageSelector) -> int:
    consumed = 0
    while True:
        pending = context.messages.peek(selector)
        if pending is None:
            return consumed
        metadata = pending.message.metadata if isinstance(pending.message.metadata, dict) else {}
        if str(metadata.get("source") or "") in {"scheduled_task", "event_automation"}:
            # Automation occurrences must be processed as independent tasks so
            # their delivery metadata survives through outbox projection.
            return consumed
        disposition = str(metadata.get("routing_disposition") or "pdca_task")
        if disposition not in {"steering", "correlated_response"}:
            return consumed
        queued = context.consume_message(selector)
        if queued is None:
            return consumed
        task.append_conversation_message(queued.message.user, queued.message.prompt)
        if queued.message.user == task.user:
            task.merge_attachments(queued.message.metadata)
        consumed += 1


def _markdown_is_empty(value: str) -> bool:
    rendered = str(value or "").strip()
    return not rendered or rendered == "- (none)"
