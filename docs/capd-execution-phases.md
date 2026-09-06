# CAPD execution phases

## Implemented prompt contract

Plan selects one bounded execution phase: either one direct tool call or a Python
program containing related calls, dependencies, explicit branches, and bounded
loops. The horizon ends when the next action requires fresh model judgment.
Batching dependent calls can save model round trips even when those calls execute
sequentially. Independent calls may run concurrently only when their tool
descriptors explicitly mark them read-only.

Plan receives the runner's availability and each exposed tool's read-only flag.
Program availability is advisory: execution must still handle runtime failures.
The existing Docker verification checks Python startup, not end-to-end bridge
communication, and must be strengthened separately.

Do executes the selected phase through the existing runtime. Programs must inspect
tool status envelopes and tool-specific results, stop dependent work on unexpected
failure, and return evidence and partial failures. Check assesses the recorded child
calls as well as the program result; a successful program exit is not proof of task
success. These prompt changes do not change graph routing or runtime enforcement.

## Proposed Act responsibility (not implemented)

Check answers: what outcomes does the execution evidence support, and what remains
unmet? Act answers: does the current approach still make sense?

Act should preserve the user's desired outcomes while choosing among continuing,
revising the approach, requesting a necessary decision, and finishing. Revising an
approach does not authorize silently relaxing acceptance criteria. When alternatives
change scope, tradeoffs, or required authorization, Act should describe concrete
options and request the user's choice. Routine recoverable failures need not cause
a confirmation request.

Keep straightforward continuation and completion deterministic. Invoke a strategic
assessment when evidence reveals repeated failure, no progress, an invalidated
assumption, or a decision that requires the user. Avoid a mandatory extra inference
call on every successful phase.

Act's structured decision would feed Plan; Plan would select the existing question
or response tool, and Do would execute it. Waiting for a required answer would use
the existing task-parking mechanism. Act should not execute tools itself.

This requires extending Check beyond its current checkbox-only output with
evidence references and explicit blockers, defining Act's decision schema, and
testing route transitions and preservation of unmet outcomes. Merely changing the
Act prompt cannot implement these transitions in the current graph.

## Validation before measuring gains

Repair and test the container-to-host tool bridge end to end. Then compare direct
and program phases on the same representative tasks, measuring successful child
calls per phase, model-call counts, execution failures, time to visible answer,
total latency, and task correctness. Saved checkpoints can repeat plan entries and
are not a complete execution log; they cannot establish aggregate time savings.
