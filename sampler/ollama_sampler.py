import time
from typing import Any

import ollama
from dotenv import load_dotenv
from ..types_eval import MessageList, SamplerBase, SamplerResponse

OLLAMA_SYSTEM_MESSAGE_DEFAULT = "You are a helpful assistant."


class OllamaSampler(SamplerBase):
    """
    Sample from Ollama's chat completion API
    """

    def __init__(
        self,
        model: str = "qwen3:4b",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        load_dotenv()
        self.model = model
        self.system_message = system_message or OLLAMA_SYSTEM_MESSAGE_DEFAULT
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=message_list,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                )
                content = response["message"]["content"]
                if not content:
                    raise ValueError("Ollama API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # exponential back off
                print(
                    f"Exception, waiting and retrying {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
