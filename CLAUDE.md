# AgentScope – AI Assistant Guide (CLAUDE.md)

AgentScope is a flexible, production-ready multi-agent framework (v1.0.18) by Alibaba Tongyi Lab. It supports building, running, and evaluating multi-agent pipelines with integrations for major LLM providers, RAG, memory, MCP, A2A, real-time voice, and model tuning.

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
│   ├── memory/          # InMemoryMemory, RedisMemory, LongTermMemory backends
│   ├── pipeline/        # ChatRoom, MsgHub, SequentialPipeline, FanoutPipeline
│   ├── session/         # JSONSession, RedisSession
│   ├── rag/             # Readers, vector stores, knowledge bases
│   ├── evaluate/        # Evaluation framework and benchmarks
│   ├── tracing/         # OpenTelemetry integration
│   ├── realtime/        # WebSocket-based real-time voice agents
│   ├── tts/             # Text-to-Speech model integrations
│   ├── mcp/             # Model Context Protocol clients
│   ├── a2a/             # Agent-to-Agent protocol
│   ├── tuner/           # DSPy / Trinity-RFT model and prompt tuning
│   ├── embedding/       # Embedding model integrations and caching
│   ├── token/           # Token counting per provider
│   ├── hooks/           # Hook system (pre/post reply/observe/print)
│   ├── plan/            # Plan management and notebook utilities
│   ├── module/          # StateModule for state management
│   ├── types/           # Type definitions and protocols
│   ├── exception/       # Custom exceptions
│   ├── tune/            # Placeholder module
│   └── _utils/          # Internal utilities
├── tests/               # 46 unit test files
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

**Core dependencies** (may be imported at module top level): `aiofiles`, `aioitertools`, `anthropic`, `dashscope`, `docstring_parser`, `filetype`, `json5`, `json_repair`, `mcp>=1.13`, `numpy`, `openai`, `opentelemetry-*`, `python-datauri`, `python-frontmatter`, `python-socketio`, `shortuuid`, `sounddevice`, `sqlalchemy`, `tiktoken`.

**Optional dependency groups:** `a2a`, `realtime`, `models` (ollama, gemini), `tokens` (transformers, jinja2), `redis_memory`, `mem0ai`, `reme`, `memory`, `readers`, `vdbs`, `rag`, `evaluate`, `tuner`, `full`.

---

## Pre-commit Checks

The pipeline enforces: `black` (79-char line limit), `flake8`, `pylint`, `mypy` (strict), `check-yaml/json/toml`, `detect-private-key`, `add-trailing-comma`, `pyroma`.

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

`Msg` is the core message object. `content` can be a plain string or a list of typed `ContentBlock` objects:

```python
from agentscope.message import Msg

msg = Msg(name="Alice", content="Hello!", role="user")
# Multi-modal content using ContentBlocks:
from agentscope.message import TextBlock, ImageBlock, URLSource
msg = Msg(name="Alice", role="user", content=[
    TextBlock(text="What's in this image?"),
    ImageBlock(image=URLSource(url="https://example.com/img.png")),
])
```

Available block types: `TextBlock`, `ThinkingBlock`, `ToolUseBlock`, `ToolResultBlock`, `ImageBlock`, `AudioBlock`, `VideoBlock`.
Available sources: `Base64Source`, `URLSource`.

### Agent Lifecycle

Agents extend `AgentBase` and implement `reply()`. Hooks attach behavior at lifecycle points:

```python
class MyAgent(AgentBase):
    async def reply(self, x: Msg | None = None) -> Msg:
        # process input, return output Msg
        ...
```

Hook points: `pre_reply`, `post_reply`, `pre_observe`, `post_observe`, `pre_print`, `post_print`.

**Built-in agents:**
- `ReActAgent` / `ReActAgentBase` – tool-calling ReAct loop
- `UserAgent` – wraps `TerminalUserInput` or `StudioUserInput`
- `A2AAgent` – Agent-to-Agent protocol
- `RealtimeAgent` – WebSocket-based real-time voice

### Tool / Toolkit

```python
from agentscope.tool import Toolkit

toolkit = Toolkit()

@toolkit.tool
def my_tool(x: int) -> str:
    """Tool docstring shown to the model."""
    return str(x)
```

Built-in tools available: `execute_python_code`, `execute_shell_command`, `view_text_file`, `write_text_file`, `insert_text_file`, plus DashScope and OpenAI multimodal helpers (image, audio, video generation/transcription).

### Model Integration

Models live in `src/agentscope/model/`. Each provider has a paired formatter in `src/agentscope/formatter/`. When adding a new provider:
1. Create `_my_provider_model.py` in `model/`
2. Create `_my_provider_formatter.py` in `formatter/`
3. Export from the respective `__init__.py`
4. Add any required optional deps to `pyproject.toml`

**Supported providers:** `OpenAIChatModel`, `AnthropicChatModel`, `GeminiChatModel`, `DashScopeChatModel`, `OllamaChatModel`, `TrinityChatModel`.

**Formatters:** OpenAI, Anthropic, Gemini, DashScope, Ollama, DeepSeek (each with `ChatFormatter` and `MultiAgentFormatter` variants), plus `A2AChatFormatter`.

### Memory

```python
from agentscope.memory import InMemoryMemory

mem = InMemoryMemory()
mem.add(msg)
history = mem.get_messages()
```

**Working memory backends:** `InMemoryMemory`, `RedisMemory`, `AsyncSQLAlchemyMemory`.

**Long-term memory backends:** `Mem0LongTermMemory`, `ReMePersonalLongTermMemory`, `ReMeTaskLongTermMemory`, `ReMeToolLongTermMemory` (require respective optional deps).

### Pipeline / Multi-Agent

```python
from agentscope.pipeline import MsgHub, ChatRoom, sequential_pipeline, fanout_pipeline

# Functional pipeline
result = await sequential_pipeline([agent_a, agent_b], initial_msg)
results = await fanout_pipeline([agent_a, agent_b], initial_msg)

# Class-based
pipeline = SequentialPipeline([agent_a, agent_b])
```

### MCP Client

```python
from agentscope.mcp import StdIOStatefulClient, HttpStatelessClient, HttpStatefulClient
```

Clients: `StdIOStatefulClient` (subprocess), `HttpStatelessClient`, `HttpStatefulClient`.

### Plan / Notebook

```python
from agentscope.plan import Plan, SubTask, PlanNotebook, InMemoryPlanStorage
```

### State Module

```python
from agentscope.module import StateModule
```

---

## Module Quick Reference

| Module | Key Exports |
|---|---|
| `agentscope.agent` | `AgentBase`, `ReActAgentBase`, `ReActAgent`, `UserAgent`, `A2AAgent`, `RealtimeAgent`, `TerminalUserInput`, `StudioUserInput` |
| `agentscope.message` | `Msg`, `TextBlock`, `ThinkingBlock`, `ToolUseBlock`, `ToolResultBlock`, `ImageBlock`, `AudioBlock`, `VideoBlock`, `Base64Source`, `URLSource`, `ContentBlock` |
| `agentscope.model` | `ChatModelBase`, `ChatResponse`, `OpenAIChatModel`, `AnthropicChatModel`, `GeminiChatModel`, `DashScopeChatModel`, `OllamaChatModel`, `TrinityChatModel` |
| `agentscope.formatter` | `FormatterBase`, `TruncatedFormatterBase`, plus Chat/MultiAgent formatters for OpenAI, Anthropic, Gemini, DashScope, Ollama, DeepSeek, A2A |
| `agentscope.tool` | `Toolkit`, `ToolResponse`, code/file/multimodal built-in tools |
| `agentscope.memory` | `MemoryBase`, `InMemoryMemory`, `RedisMemory`, `AsyncSQLAlchemyMemory`, `LongTermMemoryBase`, `Mem0LongTermMemory`, `ReMe*LongTermMemory` |
| `agentscope.pipeline` | `MsgHub`, `ChatRoom`, `SequentialPipeline`, `sequential_pipeline`, `FanoutPipeline`, `fanout_pipeline`, `stream_printing_messages` |
| `agentscope.session` | `SessionBase`, `JSONSession`, `RedisSession` |
| `agentscope.rag` | `Document`, `DocMetadata`, readers (`TextReader`, `PDFReader`, `ImageReader`, `WordReader`, `ExcelReader`, `PowerPointReader`), stores (`QdrantStore`, `MilvusLiteStore`, `OceanBaseStore`, `MongoDBStore`, `AlibabaCloudMySQLStore`), `KnowledgeBase`, `SimpleKnowledge` |
| `agentscope.evaluate` | `EvaluatorBase`, `RayEvaluator`, `GeneralEvaluator`, `MetricBase`, `MetricResult`, benchmarks (`ACEBenchmark`, `ACEAccuracy`, `ACEProcessAccuracy`, `ACEPhone`), `Task`, `SolutionOutput` |
| `agentscope.tracing` | `setup_tracing`, `trace`, `trace_llm`, `trace_reply`, `trace_format`, `trace_toolkit`, `trace_embedding` |
| `agentscope.realtime` | `RealtimeModelBase`, `DashScopeRealtimeModel`, `OpenAIRealtimeModel`, `GeminiRealtimeModel`, event types |
| `agentscope.tts` | `TTSModelBase`, `TTSResponse`, `TTSUsage`, `DashScopeTTSModel`, `DashScopeRealtimeTTSModel`, `GeminiTTSModel`, `OpenAITTSModel`, CosyVoice variants |
| `agentscope.mcp` | `MCPToolFunction`, `MCPClientBase`, `StatefulClientBase`, `StdIOStatefulClient`, `HttpStatelessClient`, `HttpStatefulClient` |
| `agentscope.a2a` | `AgentCardResolverBase`, `FileAgentCardResolver`, `WellKnownAgentCardResolver`, `NacosAgentCardResolver` |
| `agentscope.tuner` | `tune`, `select_model`, `tune_prompt`, config/result/validator types |
| `agentscope.embedding` | `EmbeddingModelBase`, `EmbeddingResponse`, `EmbeddingUsage`, DashScope/OpenAI/Gemini/Ollama text/multimodal embeddings, `FileEmbeddingCache` |
| `agentscope.token` | `TokenCounterBase`, `CharTokenCounter`, `GeminiTokenCounter`, `OpenAITokenCounter`, `AnthropicTokenCounter`, `HuggingFaceTokenCounter` |
| `agentscope.hooks` | `as_studio_forward_message_pre_print_hook` |
| `agentscope.plan` | `SubTask`, `Plan`, `DefaultPlanToHint`, `PlanNotebook`, `PlanStorageBase`, `InMemoryPlanStorage` |
| `agentscope.module` | `StateModule` |
| `agentscope.types` | `AgentHookTypes`, `ReActAgentHookTypes`, `Embedding`, `JSONPrimitive`, `JSONSerializableObject`, `ToolFunction` |
| `agentscope.exception` | `AgentOrientedExceptionBase`, `ToolInterruptedError`, `ToolNotFoundError`, `ToolInvalidArgumentsError` |

---

## Examples Directory

```
examples/
├── agent/           # Specialized agents (deep_research, voice, react, a2a, realtime, browser)
├── functionality/   # Feature showcases (memory, RAG, MCP, TTS, structured output, streaming)
├── workflows/       # Multi-agent patterns (concurrent, conversation, debate, realtime)
├── evaluation/      # ACE benchmark
├── deployment/      # Planning agent deployment
├── game/            # Werewolves game
├── tuner/           # Model tuning, prompt tuning, model selection
└── integration/     # AlibabaCloud MCP, Qwen deep research
```

New agent examples belong in `examples/`, not in `src/agentscope/agent/` — the core library maintains only `ReActAgent` and protocol agents.

---

## What to Avoid

- Importing optional packages at module top level (breaks users who haven't installed that extra).
- Creating new public files without the `_` prefix in `src/agentscope/`.
- Exposing symbols without adding them to `__init__.py` and `__all__`.
- File-wide lint/type suppression comments.
- Adding logic without corresponding unit tests in `tests/`.
- Hardcoding credentials or model names that should be configurable.
- Amending published commits; always create new commits.
- Adding optional deps to the core `dependencies` list in `pyproject.toml`.
