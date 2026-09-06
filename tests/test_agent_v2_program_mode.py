from __future__ import annotations

import json
from typing import Any

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.intelligence.pdca.nodes import do_node, plan_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.programs import ProgramRunner
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService


def test_plan_node_accepts_python_program_execution(monkeypatch) -> None:
    import importlib

    module = importlib.import_module("alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node")
    monkeypatch.setattr(
        module,
        "_call_tool_planning_llm",
        lambda _prompt: {
            "execution_mode": "program",
            "program": {"language": "python", "source": "async def main(tools):\n    return {'ok': True}"},
            "internal_state": "Comparing the available information.",
        },
    )
    task = TaskState(goal="Compare options", acceptance_criteria_md="1.- [ ] Compared")
    plan_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=_Registry(), program_runner=_Runner("success")))
    planned = json.loads(task.plan_json)[0]
    assert planned["execution_mode"] == "program"
    assert planned["program"]["language"] == "python"


def test_program_mode_records_child_tool_calls_and_final_json() -> None:
    task = _program_task()
    registry = _Registry()
    runner = _Runner("success")
    do_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, program_runner=runner))
    execution = json.loads(task.plan_json)[0]["execution"]
    assert execution["status"] == "success"
    assert execution["result"]["program_result"] == {"answer": 42}
    assert execution["result"]["tool_calls"][0]["tool_id"] == "native.read"
    assert registry.calls == [("native.read", {"value": 42})]


def test_program_mode_waiting_result_parks_task() -> None:
    task = _program_task()
    do_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=_Registry(), program_runner=_Runner("waiting")))
    assert task.status == "waiting_user"
    assert json.loads(task.plan_json)[0]["execution"]["status"] == "waiting"


def test_programmatic_parallel_call_rejects_non_read_only_tool() -> None:
    registry = _Registry()
    service = ToolInvocationService(context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry), task=TaskState())
    result = service.invoke("native.write", {}, parallel=True)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "tool_not_parallel_safe"


def test_plan_node_rejects_program_mode_when_docker_runner_is_unavailable(monkeypatch) -> None:
    import importlib

    module = importlib.import_module("alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node")
    monkeypatch.setattr(module, "_call_tool_planning_llm", lambda _prompt: _program_value())
    task = TaskState(goal="Compare", acceptance_criteria_md="1.- [ ] Compared")
    plan_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=_Registry(), program_runner=_UnavailableRunner()))
    assert "planned_tool_call" not in task.metadata
    assert "Program mode is unavailable" in task.metadata["tool_call_plan_prompt"]


def test_planner_receives_program_availability_and_tool_parallel_eligibility(monkeypatch) -> None:
    import importlib

    module = importlib.import_module("alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node")
    captured = {}

    def plan(prompt):
        captured["prompt"] = prompt
        return _program_value()

    monkeypatch.setattr(module, "_call_tool_planning_llm", plan)
    task = TaskState(goal="Compare", acceptance_criteria_md="1.- [ ] Compared")
    plan_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=_Registry(), program_runner=_Runner("success")))

    prompt = captured["prompt"]
    assert "Program mode is available" in prompt
    read_line = next(line for line in prompt.splitlines() if line.startswith("- native.read |"))
    write_line = next(line for line in prompt.splitlines() if line.startswith("- native.write |"))
    assert "read_only=true" in read_line
    assert "read_only=false" in write_line
    assert "tools.gather` only for independent tools explicitly marked `read_only=true" in prompt


def test_docker_runner_reports_unavailability_without_local_fallback(monkeypatch) -> None:
    runner = ProgramRunner(docker_bin="missing-docker")
    monkeypatch.setattr(runner, "available", lambda: False)
    outcome = runner.run(source="async def main(tools):\n    return {}", invocation_service=object())
    assert outcome["status"] == "failed"
    assert outcome["error"]["code"] == "docker_unavailable"


def _program_task() -> TaskState:
    task = TaskState(goal="Program", acceptance_criteria_md="1.- [ ] Done")
    task.append_plan_call(
        {
            "id": "program-1",
            "execution_mode": "program",
            "program": {"language": "python", "source": "async def main(tools):\n    return {}"},
            "internal_state": "Running a bounded program.",
        }
    )
    return task


class _Registry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.descriptors = {
            "native.read": ToolDescriptor("native.read", "read", ToolKind.NATIVE, read_only=True),
            "native.write": ToolDescriptor("native.write", "write", ToolKind.NATIVE),
        }

    def get(self, tool_id: str) -> ToolDescriptor | None:
        return self.descriptors.get(tool_id)

    def list(self) -> tuple[ToolDescriptor, ...]:
        return tuple(self.descriptors.values())

    def execute(self, tool_id: str, arguments: dict[str, Any], execution_context=None) -> dict[str, Any]:
        _ = execution_context
        self.calls.append((tool_id, dict(arguments)))
        return {"value": arguments.get("value")}


class _Runner:
    def __init__(self, status: str) -> None:
        self.status = status

    def run(self, *, source: str, invocation_service: ToolInvocationService) -> dict[str, Any]:
        assert "async def main" in source
        if self.status == "waiting":
            return {"status": "waiting", "program_result": {"question": "q"}, "tool_calls": []}
        child = invocation_service.invoke("native.read", {"value": 42})
        return {"status": "success", "program_result": {"answer": 42}, "tool_calls": [child]}

    def available(self) -> bool:
        return True


class _UnavailableRunner:
    def available(self) -> bool:
        return False


def _program_value() -> dict[str, Any]:
    return {
        "execution_mode": "program",
        "program": {"language": "python", "source": "async def main(tools):\n    return {}"},
        "internal_state": "Compare options.",
    }
