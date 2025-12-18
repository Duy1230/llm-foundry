# Implementation Plan: LLM-Foundry

This document outlines the step-by-step implementation strategy for the `llm-foundry` project. The goal is to build a high-performance, modular training framework leveraging state-of-the-art (SOTA) libraries like **Unsloth** for 2x faster training/memory efficiency, **HuggingFace TRL** for training loops, and **Accelerate** for hardware abstraction.

## Phase 1: Environment & Project Skeleton
**Goal:** Set up the directory structure and dependency management.

1.  **Directory Setup**:
    -   Create the root `llm-foundry/` and all subdirectories (`configs`, `data`, `models`, `engine`, `utils`, `scripts`) as defined in the user structure.
    -   Create `__init__.py` files in all Python modules to make them importable.

2.  **Dependency Management (`requirements.txt`)**:
    -   **Core:** `torch`, `xformers` (if compatible).
    -   **Optimization:** `unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git` (or specific install for local), `bitsandbytes`.
    -   **HuggingFace:** `transformers`, `datasets`, `accelerate`, `peft`, `trl`.
    -   **Ops/Logging:** `wandb`, `hydra-core` (or `pyyaml` for simple configs).
    -   **Eval:** `lm_eval` (lm-evaluation-harness).

## Phase 2: Configuration System (`configs/`)
**Goal:** Create a robust YAML configuration schema that controls every aspect of the pipeline.

1.  **Define Configuration Classes**:
    -   Use `dataclasses` or `pydantic` to define expected config structures in a new helper file (e.g., `utils/config_schema.py`).
    -   **Keys:** `ModelConfig`, `TokenizerConfig`, `DataConfig`, `TrainingConfig` (hyperparams), `LoRAConfig`.

2.  **Create YAML Templates**:
    -   `configs/cpt/llama3_cpt.yaml`: Settings for continued pre-training (raw text, lower LR).
    -   `configs/sft/mistral_chat.yaml`: Settings for instruction tuning (chat templates, higher LR, LoRA rank/alpha).

## Phase 3: Data Pipeline (`data/`)
**Goal:** Efficiently load, tokenize, and pack data using Streaming to handle massive datasets.

1.  **Processors (`data/processors.py`)**:
    -   Implement `DataProcessor`:
        -   Logic to apply Chat Templates (`tokenizer.apply_chat_template`).
        -   Tokenization logic.
        -   **Key Feature:** Ensure compatibility with `unsloth`'s fast tokenizer.

2.  **Loaders (`data/loaders.py`)**:
    -   Implement `load_streaming_dataset`:
        -   Wrap `datasets.load_dataset` with `streaming=True`.
        -   Implement an `IterableDataset` wrapper to shuffle buffer and interleave datasets if mixing sources.

3.  **Collators (`data/collators.py`)**:
    -   Implement `DataCollatorForCompletionOnlyLM` (via `trl`) for masking user prompts in SFT.
    -   Implement **Sequence Packing**:
        -   If using `unsloth`, check for native packing support.
        -   Otherwise, implement a `PackedDataset` logic that concatenates samples to `max_seq_length` to reduce padding efficiency loss.

## Phase 4: Model & Adapter Logic (`models/`)
**Goal:** Integrate Unsloth for ultra-fast loading and LoRA injection.

1.  **Registry (`models/registry.py`)**:
    -   Implement `ModelLoader`:
        -   **Primary Path:** Use `unsloth.FastLanguageModel.from_pretrained`.
        -   **Fallback Path:** Use `transformers.AutoModelForCausalLM` (for models Unsloth doesn't support yet).
        -   Handle 4-bit/8-bit quantization flags (`load_in_4bit=True`).

2.  **Adapters (`models/adapters.py`)**:
    -   Implement `AdapterManager`:
        -   Use `FastLanguageModel.get_peft_model` for Unsloth path.
        -   Map config YAML keys (r, lora_alpha, target_modules) to the function arguments.
        -   Ensure `gradient_checkpointing` is enabled correctly ("unsloth" handles this automatically usually).

## Phase 5: Core Engine (`engine/`)
**Goal:** Abstract the training loop, utilizing `TRL`'s `SFTTrainer` for best practices.

1.  **Objectives (`engine/objectives.py`)**:
    -   Define custom loss functions if needed (e.g., `NEFTune` noise embeddings can be passed to Trainer).
    -   *Note: Unsloth handles CrossEntropy optimization natively.*

2.  **Callbacks (`engine/callbacks.py`)**:
    -   Implement `WandbCallback`: Custom logging (sample generation during training to monitor quality).
    -   Implement `ProfilingCallback`: Simple timing for throughput (tokens/sec).

3.  **Trainer (`engine/trainer.py`)**:
    -   Class `CustomTrainer`:
        -   Inherit from `trl.SFTTrainer`.
        -   **Integration:** Pass the Unsloth model and tokenizer.
        -   **Optimization:** Ensure `packing=True` (if supported by config) and `dataset_num_proc`.
        -   Override `compute_loss` only if strictly necessary; otherwise rely on TRL/Unsloth.

## Phase 6: Utilities (`utils/`)
**Goal:** Helper functions for distributed training and evaluation.

1.  **Distributed (`utils/distributed.py`)**:
    -   Setup `accelerate` config parsing (checking for FSDP vs DDP).
    -   *Note: Unsloth is optimized for single GPU, but supports DDP. Ensure context managers are set correctly.*

2.  **Checkpointer (`utils/checkpointer.py`)**:
    -   Logic to save PEFT adapters separately from base model.
    -   Implement "keep last N checkpoints" logic.

3.  **Evaluation (`utils/eval.py`)**:
    -   Wrapper around `lm_eval.simple_evaluate`.
    -   Function to run benchmarks (MMLU, GSM8K) immediately after training finishes.

## Phase 7: Scripts & Entry Points (`scripts/`)
**Goal:** The user-facing CLI.

1.  **Launch Training (`scripts/launch_train.py`)**:
    -   Use `argparse` to accept `--config_path`.
    -   **Workflow:**
        1.  Parse Config.
        2.  Initialize `ModelLoader` (Unsloth).
        3.  Initialize `DataProcessor` & `Loaders`.
        4.  Setup `CustomTrainer`.
        5.  `trainer.train()`.
        6.  `trainer.save_model()`.

2.  **Merge Adapters (`scripts/merge_adapter.py`)**:
    -   Script to load base model + LoRA.
    -   Use `model.merge_and_unload()`.
    -   Save to GGUF or SafeTensors format for deployment.

## Phase 8: Testing & Verification
1.  **Sanity Check:** Run a tiny training run (10 steps) on a CPU or T4 GPU to verify pipeline flow.
2.  **Loss verification:** Ensure loss decreases on a standard dataset (Alpaca or OpenOrca).
3.  **Unsloth Speed Benchmark:** Compare iteration time vs standard HuggingFace implementation.
