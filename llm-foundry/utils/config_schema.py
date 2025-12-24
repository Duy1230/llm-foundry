"""Configuration schema definitions for LLM training."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Model configuration."""
    name_or_path: str  # HuggingFace model ID or local path
    trust_remote_code: bool = False
    use_flash_attention_2: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: str = "float16"  # float16, bfloat16, float32


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""
    name_or_path: Optional[str] = None  # If None, uses model's tokenizer
    trust_remote_code: bool = False
    padding_side: str = "right"
    truncation_side: str = "right"


@dataclass
class DataConfig:
    """Data loading configuration."""
    dataset_name: Optional[str] = None  # HuggingFace dataset name
    dataset_path: Optional[str] = None  # Local path or HuggingFace dataset path
    dataset_config_name: Optional[str] = None
    text_column: Optional[str] = None  # For CPT: column name with raw text
    instruction_column: Optional[str] = None  # For SFT: instruction column
    input_column: Optional[str] = None  # For SFT: input column (optional)
    output_column: Optional[str] = None  # For SFT: output/target column
    streaming: bool = True
    shuffle_buffer_size: int = 10000
    max_seq_length: int = 2048
    dataset_text_field: str = "text"  # Default field name for CPT
    chat_template: Optional[str] = None  # For SFT: template name (e.g., "mistral", "llama3")


@dataclass
class LoRAConfig:
    """LoRA/QLoRA adapter configuration."""
    r: int = 16
    lora_alpha: int = 32
    target_modules: Optional[List[str]] = None  # If None, auto-detect
    lora_dropout: float = 0.05
    bias: str = "none"  # none, all, lora_only
    task_type: str = "CAUSAL_LM"
    use_gradient_checkpointing: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    output_dir: str = "./outputs"
    num_train_epochs: Optional[float] = None
    max_steps: Optional[int] = None
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"  # linear, cosine, constant, etc.
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: Optional[int] = None
    save_total_limit: int = 3
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    report_to: str = "wandb"  # wandb, tensorboard, none
    run_name: Optional[str] = None
    seed: int = 42
    # Sequence packing
    packing: bool = False  # Enable sequence packing for efficiency
    # NEFTune noise
    neftune_noise_alpha: Optional[float] = None  # If set, enables NEFTune


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig
    lora: Optional[LoRAConfig] = None
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            tokenizer=TokenizerConfig(**config_dict.get("tokenizer", {})),
            data=DataConfig(**config_dict.get("data", {})),
            lora=LoRAConfig(**config_dict["lora"]) if config_dict.get("lora") else None,
            training=TrainingConfig(**config_dict.get("training", {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model.__dict__,
            "tokenizer": self.tokenizer.__dict__,
            "data": self.data.__dict__,
            "lora": self.lora.__dict__ if self.lora else None,
            "training": self.training.__dict__
        }
