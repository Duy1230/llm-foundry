llm-foundry/
├── configs/                  # YAML configurations for different stages
│   ├── cpt/                  # Continued Pre-training configs
│   │   └── llama3_cpt.yaml
│   └── sft/                  # Fine-tuning configs
│       └── mistral_chat.yaml
├── data/                     # Data processing pipelines
│   ├── __init__.py
│   ├── loaders.py            # Streaming logic (e.g., load_dataset with streaming=True)
│   ├── collators.py          # Dynamic padding & Sample Packing logic 
│   └── processors.py         # Chat template application & tokenization
├── models/                   # Model definitions & PEFT wrappers
│   ├── registry.py           # AutoModel loading logic
│   └── adapters.py           # LoRA/QLoRA injection using PEFT library
├── engine/                   # Core training logic
│   ├── trainer.py            # Abstracted training loop (integrates Accelerate/DeepSpeed)
│   ├── objectives.py         # Loss functions (CrossEntropy, NEFTune noise, DPO loss)
│   └── callbacks.py          # Logging (WandB) and profiling hooks
├── utils/
│   ├── distributed.py        # FSDP/DeepSpeed setup helpers 
│   ├── checkpointer.py       # Sharded checkpoint saving/loading management
│   └── eval.py               # Integration with lm-evaluation-harness
└── scripts/
    ├── launch_train.py       # Entry point parsing YAML config
    └── merge_adapter.py      # Post-training utility to merge LoRA weights
