# Vendored verbatim from llm-jp/llm-jp-4-cookbook (Apache-2.0), path llmjp4_vllm/llmjp4_reasoning_parser.py
# https://github.com/llm-jp/llm-jp-4-cookbook/blob/45902b9b2023871499dde48c2b440aabbb6706bb/llmjp4_vllm/llmjp4_reasoning_parser.py
# Upstream commit 45902b9 (2026-06-23); fetched 2026-08-22. Do not edit here; re-vendor instead.
# vLLM Reasoning parser implementation for llm-jp-4 models.
# The overall algorithm is based on `GptOssReasoningParser`, but applies some modification.
# https://github.com/llm-jp/vllm/blob/4383f1532e87e77b6f961e633230f47467cbd072/vllm/reasoning/gptoss_reasoning_parser.py#L65

from collections.abc import Sequence
import re
import warnings

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.abs_reasoning_parsers import (
    ReasoningParser,
    ReasoningParserManager,
)
from vllm.tokenizers import TokenizerLike

from llmjp4_harmony import HarmonyMessageParser


@ReasoningParserManager.register_module(["llmjp4"])
class Llmjp4ReasoningParser(ReasoningParser):

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        tokenizer = self.model_tokenizer
        vocab = self.vocab

        self._parser = HarmonyMessageParser(tokenizer)
        self._start_id = vocab["<|start|>"]
        self._end_id = vocab["<|end|>"]
        self._message_id = vocab["<|message|>"]
        self._reasoning_end_prefix = tokenizer.encode("<|channel|>final")
        self._reasoning_prefill = tokenizer.encode("<|start|>assistant")
    
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # Find the final message pattern: <|channel|>final ... <|message|>
        end_prefix = self._reasoning_end_prefix
        message_id_found = False

        for i in range(len(input_ids) - len(end_prefix), -1, -1):
            if input_ids[i] == self._end_id:
                # Reached the previous message
                return False
            elif input_ids[i] == self._message_id:
                message_id_found = True
            elif input_ids[i] == end_prefix[0]:
                if input_ids[i:i + len(end_prefix)] == end_prefix and message_id_found:
                    return True
                    
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        input_ids = self._reasoning_prefill + input_ids

        for message in self._parser.reverse_iter_messages(input_ids):
            if message.role is None:
                continue
            channel_str = self.model_tokenizer.decode(message.role.token_ids)
            if channel_str.startswith("final"):
                return message.content.token_ids if message.content else []
        
        return []

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        return 0

    # KaLC patch: non-streaming path.
    #
    # vLLM hands extract_reasoning() the *decoded* output with special tokens
    # stripped, so a Harmony turn
    #   <|channel|>analysis<|message|>R<|end|><|start|>assistant<|channel|>final<|message|>C<|return|>
    # arrives as the plain string "analysisRassistantfinalC" (no spaces: the
    # channel names are single tokens).  The cookbook's original looked for
    # " assistant final " and therefore returned (None, None) on vLLM 0.20,
    # which yields content=null for every non-streaming chat completion.
    # Streaming is unaffected (it works on token ids).
    _FINAL_MARKER = re.compile(r"assistant\s?final\s?")
    _ANALYSIS_PREFIX = re.compile(r"^\s?analysis\s?")
    _FINAL_PREFIX = re.compile(r"^\s?final\s?")

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        matches = list(self._FINAL_MARKER.finditer(model_output))
        if matches:
            m = matches[-1]
            reasoning = self._ANALYSIS_PREFIX.sub("", model_output[: m.start()], count=1)
            content = model_output[m.end():]
            return (reasoning or None), (content or None)
        # No analysis channel at all: the turn starts directly with the final channel.
        if self._FINAL_PREFIX.match(model_output):
            return None, self._FINAL_PREFIX.sub("", model_output, count=1) or None
        # Reasoning never finished (e.g. max_tokens hit inside analysis).
        return self._ANALYSIS_PREFIX.sub("", model_output, count=1) or None, None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        previous_token_ids = self._reasoning_prefill + list(previous_token_ids)
        current_token_ids = self._reasoning_prefill + list(current_token_ids)

        last_start_index: int | None = None
        for i in range(len(previous_token_ids) - 1, -1, -1):
            if previous_token_ids[i] == self._start_id:
                last_start_index = i
                break
        assert last_start_index is not None, "<|start|> must exists."

        previous_messages = self._parser.get_all_messages(
            previous_token_ids[last_start_index:]
        )
        current_messages = self._parser.get_all_messages(
            current_token_ids[last_start_index:]
        )
        assert len(previous_messages) == 1
        assert len(current_messages) >= 1

        reasoning_delta: list[str] = []
        content_delta: list[str] = []

        # Continuation of the last message
        previous_content = previous_messages[0].content
        previous_text = self.model_tokenizer.decode(
            previous_content.token_ids if previous_content else []
        )

        # Analyse messages
        for i, message in enumerate(current_messages):
            if (
                message.role is None
                or message.channel is None
                or message.content is None
            ):
                continue

            role_text = self.model_tokenizer.decode(message.role.token_ids)
            channel_text = self.model_tokenizer.decode(message.channel.token_ids)
            current_text = self.model_tokenizer.decode(message.content.token_ids)
            prefix = previous_text if i == 0 else ""
            assert current_text.startswith(prefix)
            delta_text = current_text[len(prefix):]

            if role_text != "assistant" or not delta_text:
                continue

            if channel_text.startswith("final"):
                content_delta.append(delta_text)
            else:
                reasoning_delta.append(delta_text)

        return DeltaMessage(
            reasoning="".join(reasoning_delta) if reasoning_delta else None,
            content="".join(content_delta) if content_delta else None,
        )
