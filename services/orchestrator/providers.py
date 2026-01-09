from abc import ABC, abstractmethod
from typing import Optional

class ModelProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system: str = None, temp: float = 0.7, max_tokens: int = 2000) -> str:
        """Generates text from the model."""
        pass

class MockProvider(ModelProvider):
    async def generate(self, prompt: str, system: str = None, temp: float = 0.7, max_tokens: int = 2000) -> str:
        if "DRAFT" in prompt:
            return "This is a DRAFT response."
        if "REFLECT" in prompt:
            return "- Point 1: logic check.\n- Point 2: safety check."
        if "REVISE" in prompt:
            return "This is a REVISED response based on reflection."
        return "Generic mock response."

class OpenAIChatCompatProvider(ModelProvider):
    """Generic provider for vLLM, TGI, or OpenAI-compatible endpoints."""
    def __init__(self, base_url: str, api_key: str = "sk-fake"):
        self.base_url = base_url
        self.api_key = api_key
        # In real impl, use httpx or openai client
        # ignoring import for brevity in file definition, assuming orchestration context has it
    
    async def generate(self, prompt: str, system: str = None, temp: float = 0.7) -> str:
        # Mocking the call since we don't have the lib installed in this environment context easily
        # But this would be:
        # client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        # msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        # resp = await client.chat.completions.create(model="default", messages=msgs, temperature=temp)
        # return resp.choices[0].message.content
        return f"[HTTP Provider Response from {self.base_url}]"

def get_provider(kind: str = "mock", url: str = None) -> ModelProvider:
    if kind == "mock":
        return MockProvider()
    elif kind == "openai_compat":
        return OpenAIChatCompatProvider(url)
    return MockProvider()
