# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)


@ReasoningParserManager.register_module("custom_cot")
class CustomCOTReasoningParser(ReasoningParser):
    """
    Reasoning parser for DeepSeek R1 model.

    The DeepSeek R1 model uses <reasoning>...</reasoning> tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.
    """

    start_token: str = "<reasoning>"
    end_token: str = "</reasoning>"

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

    def extract_reasoning_content(
            self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning content from the model output.

        For text <reasoning>abc</reasoning>xyz:
        - 'abc' goes to reasoning_content
        - 'xyz' goes to content

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Check if the start token is present in the model output, remove it
        # if it is present.
        model_output_parts = model_output.partition(self.start_token)
        model_output = model_output_parts[2] if model_output_parts[
            1] else model_output_parts[0]

        if self.end_token not in model_output:
            return None, model_output
        else:
            reasoning_content, _, content = model_output.partition(
                self.end_token)
            # If the end token is not found, return the model output as is.
            # It should not happen since we already checked for the presence
            # of the end token.
            # If generation stops right after end-of-reasoning, return null content
            final_content = content or None
            # if the final content is empty, pass the reasoning as content
            if not final_content:
                return None, reasoning_content
            return reasoning_content, final_content
