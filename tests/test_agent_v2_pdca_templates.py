from __future__ import annotations

from pathlib import Path

from jinja2 import Environment
from jinja2 import FileSystemLoader

from alphonse.agent_v2.core.intelligence.task_state import TaskState


TEMPLATE_DIR = Path("alphonse/agent_v2/core/intelligence/templates")


def test_pdca_prompt_templates_render_expected_sections() -> None:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

    for template_name in ("check_prompt.j2", "plan_prompt.j2", "act_prompt.j2"):
        rendered = env.get_template(template_name).render(
            system_prompt="System instructions",
            philosophy_md="Philosophy content",
            global_context_md="Global context",
            user_context_md="User context",
            project_context_md="Project context",
            user_prompt="User request",
            task_state_md="Task state markdown",
        )

        assert "# System Prompt" in rendered
        assert "System instructions" in rendered
        assert "## Philosophy.md" in rendered
        assert "## GlobalContext.md" in rendered
        assert "## User Context" in rendered
        assert "## Project Context" in rendered
        assert "# User Prompt" in rendered
        assert "User request" in rendered
        assert "# Task State" in rendered
        assert "Task state markdown" in rendered


def test_acceptance_criteria_prompt_template_renders_expected_sections() -> None:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

    rendered = env.get_template("acceptance_criteria_prompt.j2").render(
        check_verdict="new",
        check_reason="No acceptance criteria were present.",
        existing_acceptance_criteria_md="- (none)",
        task_state_md="Task state markdown",
    )

    assert "# System Prompt" in rendered
    assert "1.- [ ] The required outcome is true" in rendered
    assert "2.- [x] An already completed requirement is true" in rendered
    assert "## Philosophy.md" in rendered
    assert "## GlobalContext.md" in rendered
    assert "## User Context" in rendered
    assert "## Project Context" in rendered
    assert "# Check Verdict" in rendered
    assert "new" in rendered
    assert "# Existing Acceptance Criteria" in rendered
    assert "# Task State" in rendered
    assert "Task state markdown" in rendered


def test_tool_call_plan_prompt_template_renders_expected_sections() -> None:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

    rendered = env.get_template("tool_call_plan_prompt.j2").render(
        acceptance_criteria_md="1.- [ ] File exists",
        available_tools_md="- tool-1 | write_file | native | Writes files",
        task_state_md="Task state markdown",
    )

    assert "# System Prompt" in rendered
    assert "one bounded execution phase" in rendered
    assert "advance one or more acceptance criteria" in rendered
    assert "pending_silent_bash_confirmation" in rendered
    assert "next tool call must be `native.respond`" in rendered
    assert '"id": "string"' in rendered
    assert '"tool_id": "string"' in rendered
    assert '"tool_name": "string"' in rendered
    assert '"arguments": {}' in rendered
    assert '"internal_state": "short user-visible status message, 256 chars max"' in rendered
    assert "not hidden reasoning or chain-of-thought" in rendered
    assert "# Acceptance Criteria" in rendered
    assert "1.- [ ] File exists" in rendered
    assert "# Available Tools" in rendered
    assert "write_file" in rendered
    assert "# Task State" in rendered
    assert "Task state markdown" in rendered


def test_task_state_exposes_pending_silent_bash_confirmation_to_planner() -> None:
    task = TaskState(
        goal="Add sunglasses",
        metadata={
            "pending_silent_bash_confirmation": {
                "tool_call_id": "bash-call",
                "internal_state": "Added sunglasses to the TODO list.",
            }
        },
    )

    rendered = task.to_markdown_prompt()

    assert "# Required User Confirmation" in rendered
    assert "Call `native.respond` next" in rendered
    assert "Added sunglasses to the TODO list." in rendered


def test_task_state_describes_desktop_project_file_attachments() -> None:
    task = TaskState(
        goal="Analyze the attached files",
        metadata={
            "attachments": [
                {
                    "filename": "notes.txt",
                    "mime_type": "text/plain",
                    "kind": "desktop_project_file",
                    "ingestion_status": "copied",
                    "project_path": "/tmp/project/notes.txt",
                }
            ]
        },
    )

    rendered = task.to_markdown_prompt()

    assert "Name: notes.txt" in rendered
    assert "Project path: /tmp/project/notes.txt" in rendered
    assert "copied into the selected project directory" in rendered
    assert "Inspect them with normal local tools" in rendered


def test_criteria_review_prompt_template_renders_expected_sections() -> None:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

    rendered = env.get_template("criteria_review_prompt.j2").render(
        acceptance_criteria_md="1.- [ ] File exists",
        latest_executed_call_json='{"id": "plan-call-1", "execution": {"status": "success"}}',
        task_state_md="Task state markdown",
    )

    assert "# System Prompt" in rendered
    assert "Return revised acceptance criteria only" in rendered
    assert "Preserve every unmet criterion with `[ ]`" in rendered
    assert "Mark only clearly fulfilled criteria with `[x]`" in rendered
    assert "Do not decide mission success or mission failure" in rendered
    assert "# Current Acceptance Criteria" in rendered
    assert "1.- [ ] File exists" in rendered
    assert "# Latest Executed Tool Call" in rendered
    assert "plan-call-1" in rendered
    assert "# Task State" in rendered
    assert "Task state markdown" in rendered


def test_pdca_prompt_templates_have_stub_defaults() -> None:
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

    rendered = env.get_template("check_prompt.j2").render()

    assert "Stub check-node system prompt." in rendered
    assert "- (not loaded)" in rendered
    assert "- (not provided)" in rendered
