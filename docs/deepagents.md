# Deep Agents engine

Set `MOYAI_AGENT_ENGINE=deepagents` to run ordinary turns on a
[Deep Agents](https://pypi.org/project/deepagents/) graph instead of the default
Pydantic AI loop. The default is `pydantic`, which leaves behavior unchanged.

## What runs where

| Part | Deep Agents engine |
| --- | --- |
| Admission, session, history, web and Slack surfaces | Unchanged |
| Model calls | LiteLLM gateway Responses API, with the worker's key and the same per-request usage ledger |
| Computer | The session's Modal (or Lambda) computer; `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` and `execute` act on it |
| Durability | The whole graph is one Temporal activity (`moyai_deepagents_turn_v1`), single attempt, two-hour limit |
| Saved conversation | Native format, so a session can switch engines between turns |

The engine is chosen when a turn is prepared and recorded in workflow history.
Reviews, attention turns, swarm members, delegated children and turns that
resume a saved question keep the default loop.

## Not available on this engine

- Moyai capabilities other than files and shell: connected services, skills,
  memory, native planning, delegation, questions, browser and desktop tools.
  Deep Agents' own planning and `task` subagent tools are used instead.
- Live token streaming and mid-turn steering. The answer appears when the graph
  ends; guidance sent meanwhile is kept as the next turn.
- Mid-turn recovery. A worker restart fails the turn instead of replaying
  finished steps, so no command runs twice.
- Image attachments, context compaction and the per-turn usage limits.
