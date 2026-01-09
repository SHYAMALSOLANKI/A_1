from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import inspect

print(inspect.signature(LlamaRotaryEmbedding.__init__))
