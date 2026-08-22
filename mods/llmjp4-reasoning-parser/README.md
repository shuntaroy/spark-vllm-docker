# LLM-jp-4 reasoning parser

Runtime mod that makes the `llmjp4` reasoning parser from the official
[llm-jp-4-cookbook](https://github.com/llm-jp/llm-jp-4-cookbook/tree/main/llmjp4_vllm)
available to `vllm serve`, so that `llm-jp/llm-jp-4-*-thinking` models return
their Harmony-format reasoning (`<|channel|>analysis`) in `reasoning` and only
the `<|channel|>final` message in `content`.

## Why

LLM-jp-4 thinking models emit OpenAI Harmony-style channels but use a
SentencePiece tokenizer, so vLLM's built-in `openai_gptoss` parser (which relies
on the `openai_harmony` encoding) does not apply. The cookbook ships
`llmjp4_reasoning_parser.py` (registered under the name `llmjp4`) plus a
token-level helper `llmjp4_harmony.py`. The cookbook loads the parser by
`import`-ing it in a wrapper CLI (`example_cli.py`); here we load the same file
through vLLM's `--reasoning-parser-plugin` instead, which needs no wrapper.

## What it does

`run.sh`:

1. copies `llmjp4_harmony.py` into `/usr/local/lib/python3.12/dist-packages/`
   (vLLM imports the plugin file with `importlib.util.spec_from_file_location`,
   which does not put `/workspace` on `sys.path`, so the parser's bare
   `from llmjp4_harmony import HarmonyMessageParser` needs the helper to be
   importable on its own);
2. copies `llmjp4_reasoning_parser.py` to `$WORKSPACE_DIR` (= `/workspace`, the
   cwd of the eventual `vllm serve`).

Both files are vendored verbatim from cookbook commit `45902b9` (2026-06-23)
with a three-line provenance header prepended. Re-vendor rather than editing.

## How to apply

Used by `recipes/llm-jp-4-33b-thinking.yaml`. Manually:

```bash
./launch-cluster.sh --solo \
  --apply-mod mods/llmjp4-reasoning-parser \
  exec vllm serve llm-jp/llm-jp-4-33b-thinking \
    --trust-remote-code \
    --reasoning-parser-plugin /workspace/llmjp4_reasoning_parser.py \
    --reasoning-parser llmjp4
```

`--trust-remote-code` is required regardless of this mod: the model's
`tokenizer_config.json` maps `AutoTokenizer` to the bundled
`llmjp4_tokenizer.Llmjp4Tokenizer`.

Reasoning effort is selected per request via
`"chat_template_kwargs": {"reasoning_effort": "low|medium|high"}`; the chat
template defaults to `medium`.

## Compatibility with vLLM 0.20.x

Checked against the `v0.20.2` tag (the `vllm-node` image): every import the
parser uses exists there
(`vllm.entrypoints.openai.chat_completion.protocol.ChatCompletionRequest`,
`vllm.entrypoints.openai.engine.protocol.DeltaMessage` with a `reasoning`
field, `vllm.entrypoints.openai.responses.protocol.ResponsesRequest`,
`vllm.reasoning.abs_reasoning_parsers.{ReasoningParser,ReasoningParserManager}`,
`vllm.tokenizers.TokenizerLike`), and `ReasoningParser.__init__` provides the
`model_tokenizer` / `vocab` attributes the parser reads. The cookbook declares
`vllm>=0.18.0`. No adaptation was made.

Known limitation, inherited from upstream and deliberately not patched here:
`extract_reasoning()` (non-streaming `stream: false` responses) is a text-level
workaround that returns only `content` and `reasoning = None`, and emits a
`warnings.warn` on every call. Streaming responses carry both fields correctly.
See the `NOTE(odashi)` comment in the parser.

## Licence

`llmjp4_reasoning_parser.py` and `llmjp4_harmony.py` are copyright the LLM-jp
project (author Yusuke Oda) and redistributed under the Apache License 2.0, the
licence of `llm-jp/llm-jp-4-cookbook`.
