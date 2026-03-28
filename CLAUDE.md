# AgentScope – AI Assistant Guide (CLAUDE.md)

AgentScope is a flexible, production-ready multi-agent framework (v1.0.19dev) by Alibaba Tongyi Lab. It supports building, running, and evaluating multi-agent pipelines with integrations for major LLM providers, RAG, memory, MCP, A2A, real-time voice, and model tuning.

- **External docs:** https://doc.agentscope.io/
- **License:** Apache 2.0
- **Python:** 3.10+

---

## Repository Layout

```
agentscope/
├── src/agentscope/      # All source code (installable package)
│   ├── agent/           # AgentBase, ReActAgent, UserAgent, A2AAgent, RealtimeAgent
│   ├── message/         # Msg class and ContentBlock types
│   ├── model/           # LLM provider integrations
│   ├── formatter/       # Message formatters per model family
│   ├── tool/            # Toolkit, ToolResponse, built-in tools
│   ├── memory/          # WorkingMemory, LongTermMemory
│   ├── pipeline/        # ChatRoom, MessageHub, functional pipeline
│   ├── session/         # JSONSession, RedisSession
│   ├── rag/             # Readers, vector stores, knowledge bases
│   ├── evaluate/        # Evaluation framework and benchmarks
│   ├── tracing/         # OpenTelemetry integration
│   ├── realtime/        # WebSocket-based voice agents
│   ├── tts/             # Text-to-Speech integrations
│   ├── mcp/             # Model Context Protocol clients
│   ├── a2a/             # Agent-to-Agent protocol
│   ├── tuner/           # DSPy / Trinity-RFT model finetuning
│   ├── embedding/       # Embedding model integrations
│   ├── token/           # Token counting per provider
│   ├── hooks/           # Hook system (pre/post reply/observe/print)
│   ├── plan/            # Planning utilities
│   ├── exception/       # Custom exceptions
│   └── _utils/          # Internal utilities
├── tests/               # 60+ unit test files
├── examples/            # Working example scripts
├── docs/                # Tutorial source (English + Chinese)
├── .github/workflows/   # CI/CD pipelines
├── .gemini/styleguide.md  # Code review guide (mirrors these standards)
├── pyproject.toml       # Project config and all dependency groups
├── .pre-commit-config.yaml
└── CONTRIBUTING.md
```

---

## Development Setup

```bash
# Install in editable mode with all dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (required before first commit)
pre-commit install

# Run all pre-commit checks manually
pre-commit run --all-files
```

---

## Running Tests

```bash
# Run full test suite with coverage
coverage run -m pytest tests
coverage report -m

# Run a single test file
pytest tests/agent_test.py -v
```

Tests use `IsolatedAsyncioTestCase` because agents are async. CI runs on Ubuntu, Windows, macOS × Python 3.10/3.11/3.12.

---

## Code Conventions

### [MUST] Lazy Loading

Only **core dependencies** listed in `pyproject.toml` `[dependencies]` may be imported at module top level. All **optional/third-party** packages must be imported at point of use:

```python
# WRONG – optional lib at top level
import qdrant_client

# CORRECT – import inside function/method
def connect(self) -> None:
    import qdrant_client
    ...
```

For base classes from optional packages, use a factory function:

```python
def get_my_cls() -> "MyClass":
    from optional_lib import BaseClass
    class MyClass(BaseClass):
        ...
    return MyClass
```

### [MUST] File and Symbol Naming

- Every `.py` file under `src/agentscope/` must be prefixed with `_` (e.g., `_agent_base.py`).
- Public API is exposed **only** through `__init__.py` files.
- Internal helpers/classes not meant for users must also have a `_` prefix on the symbol name.

```python
# src/agentscope/agent/__init__.py
from ._agent_base import AgentBase
from ._react_agent import ReActAgent

__all__ = ["AgentBase", "ReActAgent"]
```

### [MUST] Docstrings (English only)

All classes and public methods need complete docstrings following this exact template:

```python
def func(a: str, b: int | None = None) -> str:
    """One-line summary.

    Args:
        a (`str`):
            Description of a.
        b (`int | None`, optional):
            Description of b. Defaults to None.

    Returns:
        `str`:
            Description of the return value.
    """
```

Use reStructuredText for links, notes, tips, and code blocks:

```python
"""Summary.

`Example link <https://example.com>`_

.. note:: This is a note.

.. code-block:: python

    x = 1
"""
```

### Type Hints

- All functions must be fully typed (enforced by mypy strict mode).
- Use `|` union syntax, not `Union[]`: `str | None`, `int | str`.

### [MUST] Security

- Never hardcode API keys, tokens, or passwords. Use environment variables or config files.
- Check all inputs at system boundaries for injection risks (SQL, shell, code injection).
- No debug credentials or temporary secrets committed to source.

### Code Conciseness

- Minimize unnecessary temporary variables.
- Merge duplicate logic; reuse existing utility functions from `_utils/`.
- Do not add abstractions or helpers for one-off use.

---

## Adding Dependencies

| Scenario | Where to add |
|---|---|
| Required by all users | `dependencies` in `pyproject.toml` |
| Optional feature (e.g., new vector DB) | Appropriate optional group (e.g., `[qdrant]`) |
| Dev/test only | `[dev]` group |

Never add optional dependencies to the core `dependencies` list.

---

## Pre-commit Checks

The pipeline enforces: `black` (79-char line limit), `flake8`, `pylint`, `mypy` (strict), `check-yaml/json/toml`, `detect-private-key`, `add-trailing-comma`.

- **File-level skip comments are prohibited** (e.g., `# noqa: ALL`, `# type: ignore` on a whole file).
- The only permitted skip is for agent system prompt parameters where `\n` formatting would break checks.
- Fix the underlying issue rather than suppressing the check.

---

## Git / PR Conventions

Commit messages and PR titles must follow **Conventional Commits**:

```
<type>(<scope>): <description>
```

Valid types: `feat`, `fix`, `docs`, `ci`, `refactor`, `test`, `chore`, `perf`, `style`, `build`, `revert`

- Scope is optional and must be lowercase.
- Description starts with lowercase.
- Examples:
  - `feat(memory): add redis cache support`
  - `fix(agent): resolve hook ordering issue`
  - `docs(rag): update quickstart tutorial`

PRs with "WIP" in the title skip CI. PR title format is validated automatically by GitHub Actions.

---

## Key Architectural Patterns

### Message System

`Msg` is the core message object:

```python
from agentscope.message import Msg

msg = Msg(name="Alice", content="Hello!", role="user")
# content can also be a list of ContentBlock (TextBlock, ToolUseBlock, ImageBlock, etc.)
```

### Agent Lifecycle

Agents extend `AgentBase` and implement `reply()`. Hooks attach behavior at lifecycle points:

```python
class MyAgent(AgentBase):
    async def reply(self, x: Msg | None = None) -> Msg:
        # process input, return output Msg
        ...
```

Hook points: `pre_reply`, `post_reply`, `pre_observe`, `post_observe`, `pre_print`, `post_print`.

### Tool / Toolkit

```python
from agentscope.tool import Toolkit

toolkit = Toolkit()

@toolkit.tool
def my_tool(x: int) -> str:
    """Tool docstring shown to the model."""
    return str(x)
```

### Model Integration

Models are in `src/agentscope/model/`. Each provider has a corresponding formatter in `src/agentscope/formatter/`. When adding a new provider:
1. Create `_my_provider_model.py` in `model/`
2. Create `_my_provider_formatter.py` in `formatter/`
3. Export from the respective `__init__.py`
4. Add any required optional deps to `pyproject.toml`

### Memory

```python
from agentscope.memory import WorkingMemory

mem = WorkingMemory()
mem.add(msg)
history = mem.get_messages()
```

Long-term memory backends: `JSONSession`, `RedisSession` (requires `redis` optional dep).

---

## Module Quick Reference

| Module | Purpose |
|---|---|
| `agentscope.agent` | Agent base classes and built-in agents |
| `agentscope.message` | `Msg`, `ContentBlock` types |
| `agentscope.model` | LLM provider clients (OpenAI, Anthropic, Gemini, DashScope, Ollama, Trinity) |
| `agentscope.formatter` | Convert `Msg` objects to provider-specific API format |
| `agentscope.tool` | `Toolkit`, `ToolResponse`, built-in tools |
| `agentscope.memory` | `WorkingMemory`, `LongTermMemory` |
| `agentscope.pipeline` | Multi-agent orchestration (`ChatRoom`, `MessageHub`) |
| `agentscope.session` | Session persistence (`JSONSession`, `RedisSession`) |
| `agentscope.rag` | RAG pipeline (readers, vector stores, knowledge base) |
| `agentscope.evaluate` | Evaluation framework and benchmarks |
| `agentscope.tracing` | OpenTelemetry tracing |
| `agentscope.realtime` | Real-time voice agent (WebSocket) |
| `agentscope.tts` | Text-to-Speech |
| `agentscope.mcp` | MCP client support |
| `agentscope.a2a` | Agent-to-Agent protocol |
| `agentscope.tuner` | Model finetuning (DSPy, Trinity-RFT) |
| `agentscope.embedding` | Embedding models |
| `agentscope.hooks` | Hook system |
| `agentscope.exception` | Custom exceptions |

---

## What to Avoid

- Importing optional packages at module top level (breaks users who haven't installed that extra).
- Creating new public files without the `_` prefix in `src/agentscope/`.
- Exposing symbols without adding them to `__init__.py` and `__all__`.
- File-wide lint/type suppression comments.
- Adding logic without corresponding unit tests in `tests/`.
- Hardcoding credentials or model names that should be configurable.
- Amending published commits; always create new commits.
