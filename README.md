# LLM-Foundry

A high-performance, modular training framework for LLM continued pre-training and fine-tuning. Built with state-of-the-art libraries including **Unsloth** for 2x faster training, **HuggingFace TRL** for training loops, and **Accelerate** for hardware abstraction.

## Features

- 🚀 **Ultra-fast training** with Unsloth optimization
- 🔧 **Flexible configuration** via YAML files
- 📊 **Streaming data loading** for handling massive datasets
- 🎯 **LoRA/QLoRA support** for efficient fine-tuning
- 📈 **WandB integration** for experiment tracking
- 🔄 **Sequence packing** for improved training efficiency
- 🧪 **Evaluation integration** with lm-evaluation-harness

## Project Structure

```
llm-foundry/
├── configs/                  # YAML configurations for different stages
│   ├── cpt/                  # Continued Pre-training configs
│   │   └── llama3_cpt.yaml
│   └── sft/                  # Fine-tuning configs
│       └── mistral_chat.yaml
├── data/                     # Data processing pipelines
│   ├── __init__.py
│   ├── loaders.py            # Streaming logic
│   ├── collators.py          # Dynamic padding & Sample Packing logic 
│   └── processors.py         # Chat template application & tokenization
├── models/                   # Model definitions & PEFT wrappers
│   ├── registry.py           # AutoModel loading logic
│   └── adapters.py           # LoRA/QLoRA injection using PEFT library
├── engine/                   # Core training logic
│   ├── trainer.py            # Abstracted training loop
│   ├── objectives.py         # Loss functions
│   └── callbacks.py          # Logging and profiling hooks
├── utils/
│   ├── config_schema.py      # Configuration schema definitions
│   ├── distributed.py        # FSDP/DeepSpeed setup helpers 
│   ├── checkpointer.py       # Sharded checkpoint saving/loading management
│   └── eval.py               # Integration with lm-evaluation-harness
└── scripts/
    ├── launch_train.py       # Entry point parsing YAML config
    └── merge_adapter.py      # Post-training utility to merge LoRA weights
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

The framework uses YAML configuration files to control all aspects of training. See `configs/` for examples:

- **CPT (Continued Pre-training)**: `configs/cpt/llama3_cpt.yaml`
- **SFT (Supervised Fine-tuning)**: `configs/sft/mistral_chat.yaml`

Configuration includes:
- Model and tokenizer settings
- Data loading parameters
- LoRA/QLoRA adapter configuration
- Training hyperparameters

## Usage

```bash
# Launch training with a config file
python scripts/launch_train.py --config_path configs/sft/mistral_chat.yaml
```

## Implementation Status

- ✅ **Phase 1**: Environment & Project Skeleton
- ✅ **Phase 2**: Configuration System
- ⏳ **Phase 3**: Data Pipeline (in progress)
- ⏳ **Phase 4**: Model & Adapter Logic
- ⏳ **Phase 5**: Core Engine
- ⏳ **Phase 6**: Utilities
- ⏳ **Phase 7**: Scripts & Entry Points
- ⏳ **Phase 8**: Testing & Verification

## License

[Add your license here]
