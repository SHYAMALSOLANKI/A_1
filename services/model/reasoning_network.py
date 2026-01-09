import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding
from transformers import LlamaConfig
import logging

logger = logging.getLogger(__name__)

class ReasoningConfig(LlamaConfig):
    def __init__(
        self,
        vocab_size=8192,
        hidden_size=512,
        intermediate_size=1376,
        num_hidden_layers=6,  # Physical layers
        num_attention_heads=8,
        num_reasoning_loops=3, # Recurrent steps (Total depth = layers * loops)
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            **kwargs
        )
        self.num_reasoning_loops = num_reasoning_loops

class RecurrentReasoningNetwork(nn.Module):
    """
    A Non-Toy Neural Architecture for Emergent Reasoning.
    
    Architecture:
    1. Input Embedding
    2. Recurrent Processing Block (Universal Transformer style)
       - Instead of a fixed depth, we have a 'Thinking Core' of N layers.
       - This Core is applied M times (loops).
       - Between loops, we can inject 'Gradient from the Future' (Feedback).
    3. Output Head
    
    This allows the model to 'ponder' on the input before emitting a token,
    increasing effective depth without increasing parameter count.
    """
    
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        self.recurrence = config.num_reasoning_loops
        
        # 1. Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # 2. The "Thinking Core" (Physical Weights)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        
        # 3. Feedback Injection Gate (For Audit Loop)
        # Allows external gradients/feedback to be projected into the latent stream
        self.feedback_gate = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 4. Final Norm & Head
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _make_causal_mask(self, batch_size, seq_len, dtype, device):
        # Optimized causal mask creation
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        return mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        feedback_embedding=None, # The "Force to do more" vector
        labels=None,
        **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        
        # RoPE Setup
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        
        # Causal Mask (Simplified)
        # For training, we act like a standard decoder
        causal_mask = self._make_causal_mask(batch_size, seq_length, hidden_states.dtype, hidden_states.device)

        # RECURRENT LOOP
        for loop_idx in range(self.recurrence):
            # Inject Feedback between loops if provided
            if feedback_embedding is not None and loop_idx > 0:
                gate = torch.sigmoid(self.feedback_gate(feedback_embedding)) 
                hidden_states = hidden_states + (feedback_embedding * gate)
            
            # Application of Physical Layers
            for decoder_layer in self.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    output_attentions=False,
                    use_cache=False,
                    cos=cos, sin=sin,
                    **kwargs
                )
                hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return {"loss": loss, "logits": logits}
