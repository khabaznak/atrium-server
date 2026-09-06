"""Plan node for the v2 PDCA graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import uuid4
from datetime import datetime, timezone

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import select_autoescape

from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRequest
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.tools.registry import ToolExposurePolicy

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext

_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"


def plan_node(task: TaskState, context: CoreLoopContext | None = None) -> TaskState:
    """Prepare one bounded execution phase without executing it."""
    if context is not None:
        context.emit_activity(
            phase=ImprovementPhase.PLAN,
            label="criteria ready",
            message="Using the latest acceptance criteria for planning.",
            progress={"acceptance_criteria": "" if task.acceptance_criteria_md == "- (none)" else task.acceptance_criteria_md},
        )
    tools = _tools_from_context(context)
    program_available = _program_mode_available(context)
    prompt = _render_tool_call_plan_prompt(
        task,
        tools,
        user_context_md=_user_context_md(task, context),
        project_context_md=_project_context_md(task, context),
        philosophy_md=_agent_prompt_md(context, "Philosophy.md"),
        global_context_md=_agent_prompt_md(context, "GlobalContext.md"),
        current_time_utc=datetime.now(timezone.utc).isoformat(),
        user_timezone=_user_timezone(task, context),
        program_available=program_available,
    )
    task.metadata["tool_call_plan_prompt"] = prompt

    planned_tool_call = _normalize_tool_call(
        _call_tool_planning_inference(prompt, task, tools, context),
        tools,
        program_available=program_available,
    )
    if planned_tool_call is None:
        task.metadata["tool_call_planning_llm_stubbed"] = True
        task.append_update("Plan produced no executable phase; inference may be stubbed or the plan invalid.")
        return task

    task.metadata["tool_call_planning_llm_stubbed"] = False
    task.metadata["planned_tool_call"] = planned_tool_call
    task.append_plan_call(planned_tool_call)
    if context is not None:
        context.record_memory_event(task, "Plan", planned_tool_call)
    if context is not None:
        context.emit_activity(
            phase=ImprovementPhase.PLAN,
            label="thinking",
            message=str(planned_tool_call.get("internal_state") or "").strip(),
        )
    task.append_update("Plan selected the next bounded execution phase.")
    return task


def _render_tool_call_plan_prompt(
    task: TaskState,
    tools: tuple[ToolDescriptor, ...],
    *,
    user_context_md: str = "",
    project_context_md: str = "",
    philosophy_md: str = "",
    global_context_md: str = "",
    current_time_utc: str = "",
    user_timezone: str = "UTC",
    program_available: bool = False,
) -> str:
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(default_for_string=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("tool_call_plan_prompt.j2")
    return template.render(
        acceptance_criteria_md=task.acceptance_criteria_md,
        available_tools_md=_render_tools_md(tools),
        user_context_md=user_context_md,
        project_context_md=project_context_md,
        philosophy_md=philosophy_md,
        global_context_md=global_context_md,
        current_time_utc=current_time_utc,
        user_timezone=user_timezone,
        program_available=program_available,
        task_state_md=task.to_markdown_prompt(),
    ).strip()


def _render_tools_md(tools: tuple[ToolDescriptor, ...]) -> str:
    if not tools:
        return "- (none)"
    return "\n".join(
        f"- {tool.tool_id} | {tool.name} | {tool.kind.value} | {tool.description or '(no description)'} | read_only={str(tool.read_only).lower()} | schema={tool.argument_schema}"
        for tool in tools
    )


def _tools_from_context(context: CoreLoopContext | None) -> tuple[ToolDescriptor, ...]:
    if context is None or context.tools is None:
        return ()
    return ToolExposurePolicy().select_tools(registry=context.tools)


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


def _user_timezone(task: TaskState, context: CoreLoopContext | None) -> str:
    if context is None or not callable(context.user_timezone_provider):
        return "UTC"
    try:
        return str(context.user_timezone_provider(str(task.user or "")) or "UTC")
    except (OSError, KeyError):
        return "UTC"


def _agent_prompt_md(context: CoreLoopContext | None, name: str) -> str:
    if context is None or context.prompts is None:
        return ""
    load = getattr(context.prompts, "load", None)
    if not callable(load):
        return ""
    return str(load(name).content or "").strip()


def _call_tool_planning_inference(
    prompt: str,
    task: TaskState,
    tools: tuple[ToolDescriptor, ...] = (),
    context: CoreLoopContext | None = None,
) -> dict[str, Any] | None:
    if context is not None and context.inference is not None:
        result = context.inference.plan_tool_call(
            InferenceRequest(
                prompt=prompt,
                purpose=InferencePurpose.TOOL_PLANNING,
                project_id=task.project_id,
                user=task.user,
                task_id=task.task_id,
                tools=tools,
            )
        )
        if result.model_profile is not None:
            task.metadata["tool_call_planning_model_profile"] = result.model_profile.profile_id
        if result.tool_call is not None:
            return dict(result.tool_call)
        if isinstance(result.json_value, dict):
            return dict(result.json_value)
    return _call_tool_planning_llm(prompt)


def _call_tool_planning_llm(prompt: str) -> dict[str, Any] | None:
    """Stub for the future one-tool-call planning LLM call."""
    _ = prompt
    return None


def _normalize_tool_call(
    value: dict[str, Any] | None,
    tools: tuple[ToolDescriptor, ...],
    *,
    program_available: bool = True,
) -> dict[str, Any] | None:
    if value is None:
        return None
    planned_id = str(value.get("id") or "").strip() or f"plan-call-{uuid4()}"
    execution_mode = str(value.get("execution_mode") or "direct").strip().lower()
    internal_state = str(value.get("internal_state") or "").strip()
    if execution_mode == "program":
        if not program_available:
            return None
        program = value.get("program")
        if not isinstance(program, dict) or not internal_state:
            return None
        language = str(program.get("language") or "python").strip().lower()
        source = str(program.get("source") or "").strip()
        if language != "python" or not source or len(source) > 100_000:
            return None
        return {
            "id": planned_id,
            "execution_mode": "program",
            "program": {"language": "python", "source": source},
            "internal_state": internal_state[:256],
        }
    if execution_mode != "direct":
        return None
    tool_id = str(value.get("tool_id") or "").strip()
    tool_name = str(value.get("tool_name") or "").strip()
    arguments = value.get("arguments")
    if not tool_id or not tool_name or not isinstance(arguments, dict) or not internal_state:
        return None
    if tools and tool_id not in {tool.tool_id for tool in tools}:
        return None
    normalized_arguments = _normalize_tool_arguments(tool_id, arguments)
    normalized = {
        "id": planned_id,
        "tool_id": tool_id,
        "tool_name": tool_name,
        "arguments": normalized_arguments,
        "internal_state": internal_state[:256],
    }
    if "execution_mode" in value:
        normalized["execution_mode"] = "direct"
    return normalized


def _normalize_tool_arguments(tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    if tool_id == "native.bash":
        _drop_codex_temp_cwd(normalized)
    return normalized


def _drop_codex_temp_cwd(arguments: dict[str, Any]) -> None:
    raw_cwd = arguments.get("cwd")
    if raw_cwd is None or str(raw_cwd).strip() == "":
        return
    cwd = Path(str(raw_cwd)).expanduser()
    if cwd.exists():
        return
    if any(str(part).startswith("alphonse-codex-") for part in cwd.parts):
        arguments.pop("cwd", None)


def _program_mode_available(context: CoreLoopContext | None) -> bool:
    runner = getattr(context, "program_runner", None) if context is not None else None
    available = getattr(runner, "available", None)
    if not callable(available):
        return False
    try:
        return bool(available())
    except (OSError, RuntimeError):
        return False
