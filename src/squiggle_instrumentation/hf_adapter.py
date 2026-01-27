"""HuggingFace model adapter for probe experiments with LoRA micro-finetuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class HFProbeAdapter:
    """
    Adapter for HuggingFace transformer models with LoRA micro-finetuning.

    Implements the ProbeAdapter protocol for use with the probe runner.
    """

    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    max_seq_length: int = 512
    torch_dtype: Optional[torch.dtype] = None
    trust_remote_code: bool = False

    # Internal state
    _model: Optional[Any] = field(default=None, repr=False)
    _tokenizer: Optional[Any] = field(default=None, repr=False)
    _peft_model: Optional[Any] = field(default=None, repr=False)
    _hooks: List[Any] = field(default_factory=list, repr=False)
    _captured_tensors: Dict[int, torch.Tensor] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path

    def _ensure_loaded(self, device: str) -> None:
        """Lazy-load the model and tokenizer."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers package required. Install with: pip install transformers"
            )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype or torch.float32,
            trust_remote_code=self.trust_remote_code,
            device_map=device if device != "cpu" else None,
        )
        if device == "cpu":
            self._model = self._model.to(device)

    def _get_layer_modules(self) -> List[nn.Module]:
        """Get the list of transformer layer modules."""
        model = self._model

        # Try common attribute names for transformer layers
        if hasattr(model, "model"):
            model = model.model
        if hasattr(model, "transformer"):
            model = model.transformer

        # Look for layers
        if hasattr(model, "layers"):
            return list(model.layers)
        elif hasattr(model, "h"):
            return list(model.h)
        elif hasattr(model, "blocks"):
            return list(model.blocks)
        elif hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            return list(model.decoder.layers)
        else:
            raise ValueError(
                f"Could not find transformer layers in model. "
                f"Model type: {type(self._model)}, available attrs: {dir(model)}"
            )

    def _register_hooks(self, layers: List[int], device: str) -> None:
        """Register forward hooks to capture layer outputs."""
        self._remove_hooks()
        self._captured_tensors.clear()

        layer_modules = self._get_layer_modules()

        for layer_idx in layers:
            if layer_idx >= len(layer_modules):
                continue

            module = layer_modules[layer_idx]

            def make_hook(idx: int) -> Callable:
                def hook(module: nn.Module, input: Any, output: Any) -> None:
                    # Handle different output formats
                    if isinstance(output, tuple):
                        tensor = output[0]
                    else:
                        tensor = output

                    # Store detached clone
                    self._captured_tensors[idx] = tensor.detach().clone()

                return hook

            handle = module.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def capture_layer_tensors(
        self, *, layers: List[int], device: str
    ) -> Dict[int, torch.Tensor]:
        """
        Capture hidden state tensors at specified layers.

        Args:
            layers: Layer indices to capture
            device: Device to run on

        Returns:
            Dict mapping layer index to captured tensor (batch, seq, hidden)
        """
        self._ensure_loaded(device)
        self._register_hooks(layers, device)

        # Run a forward pass with dummy input to capture activations
        model = self._peft_model if self._peft_model is not None else self._model

        # Create dummy input
        dummy_text = "This is a test input for capturing layer activations."
        inputs = self._tokenizer(
            dummy_text,
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            model(**inputs)

        # Get captured tensors
        result = dict(self._captured_tensors)
        self._remove_hooks()
        return result

    def micro_finetune(
        self,
        *,
        family_id: str,
        steps: int,
        lr: float,
        seed: int,
        device: str,
        training_data: Optional[List[str]] = None,
    ) -> None:
        """
        Perform LoRA micro-finetuning on family data.

        Args:
            family_id: Family identifier (for logging)
            steps: Number of training steps
            lr: Learning rate
            seed: Random seed
            device: Device to run on
            training_data: Optional list of training texts (if None, uses dummy data)
        """
        self._ensure_loaded(device)

        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("peft package required. Install with: pip install peft")

        torch.manual_seed(seed)

        # Configure LoRA
        target_modules = self.lora_target_modules
        if target_modules is None:
            # Default target modules for common architectures
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        # Apply LoRA
        if self._peft_model is None:
            self._peft_model = get_peft_model(self._model, lora_config)
        self._peft_model.train()

        # Prepare training data
        if training_data is None:
            # Use dummy data if none provided
            training_data = [
                f"Problem {i}: Solve the equation x + {i} = {i*2}. Solution: x = {i}."
                for i in range(max(steps, 10))
            ]

        # Tokenize
        def tokenize_batch(texts: List[str]) -> Dict[str, torch.Tensor]:
            encoded = self._tokenizer(
                texts,
                return_tensors="pt",
                max_length=self.max_seq_length,
                truncation=True,
                padding=True,
            )
            return {k: v.to(device) for k, v in encoded.items()}

        # Training loop
        optimizer = torch.optim.AdamW(
            self._peft_model.parameters(),
            lr=lr,
            weight_decay=0.01,
        )

        batch_size = min(4, len(training_data))

        for step in range(steps):
            # Sample batch
            batch_idx = step % (len(training_data) // batch_size + 1)
            start = batch_idx * batch_size
            end = min(start + batch_size, len(training_data))
            if start >= len(training_data):
                start = 0
                end = batch_size

            batch_texts = training_data[start:end]
            if not batch_texts:
                batch_texts = training_data[:batch_size]

            inputs = tokenize_batch(batch_texts)

            # Forward pass with labels for language modeling loss
            inputs["labels"] = inputs["input_ids"].clone()
            outputs = self._peft_model(**inputs)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._peft_model.eval()

    def get_num_layers(self) -> int:
        """Get the number of transformer layers."""
        self._ensure_loaded("cpu")
        return len(self._get_layer_modules())


def load_family_training_data(
    family_file: Path,
    max_items: int = 100,
) -> List[str]:
    """
    Load training texts from a family JSONL file.

    Args:
        family_file: Path to family JSONL file
        max_items: Maximum items to load

    Returns:
        List of training texts (problem + solution)
    """
    import json

    texts = []
    with family_file.open() as f:
        for i, line in enumerate(f):
            if i >= max_items:
                break
            item = json.loads(line)

            # Build training text from problem and solution
            problem = item.get("content", {}).get("problem", "")
            solution = item.get("provenance", {}).get("generated_solution", "")

            if problem:
                text = f"Problem: {problem}"
                if solution:
                    text += f"\n\nSolution: {solution}"
                texts.append(text)

    return texts
