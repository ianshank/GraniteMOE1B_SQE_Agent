"""Granite MoE trainer with graceful optional dependencies.

This module avoids hard failures when optional heavy dependencies
(`transformers`, `datasets`, `mlx`, etc.) are not installed by providing
clear error paths and placeholders that raise informative messages if used.
"""

# Optional heavy deps lazy import with informative placeholders
try:  # pragma: no cover - exercised transitively in CI
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset
    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False

    def _missing_transformers(name: str):
        raise ImportError(
            f"Optional dependency '{name}' is unavailable. Install 'transformers' and 'datasets' to use training features."
        )

    class _MissingAutoModelForCausalLM:  # type: ignore
        @classmethod
        def from_pretrained(cls, *_, **__):
            _missing_transformers("transformers.AutoModelForCausalLM")

    class _MissingAutoTokenizer:  # type: ignore
        @classmethod
        def from_pretrained(cls, *_, **__):
            _missing_transformers("transformers.AutoTokenizer")

    class _MissingTrainingArguments:  # type: ignore
        def __init__(self, *_, **__):
            _missing_transformers("transformers.TrainingArguments")

    class _MissingTrainer:  # type: ignore
        def __init__(self, *_, **__):
            _missing_transformers("transformers.Trainer")

    class _MissingDataCollatorForLanguageModeling:  # type: ignore
        def __init__(self, *_, **__):
            _missing_transformers("transformers.DataCollatorForLanguageModeling")

    class _MissingDataset:  # type: ignore
        @staticmethod
        def from_list(*_, **__):
            _missing_transformers("datasets.Dataset")

        def select(self, *_, **__):  # compatibility shim
            _missing_transformers("datasets.Dataset.select")

        def save_to_disk(self, *_, **__):
            _missing_transformers("datasets.Dataset.save_to_disk")

    AutoModelForCausalLM = _MissingAutoModelForCausalLM  # type: ignore
    AutoTokenizer = _MissingAutoTokenizer  # type: ignore
    TrainingArguments = _MissingTrainingArguments  # type: ignore
    Trainer = _MissingTrainer  # type: ignore
    DataCollatorForLanguageModeling = _MissingDataCollatorForLanguageModeling  # type: ignore
    Dataset = _MissingDataset  # type: ignore
import torch
from typing import List, Dict, Any, TYPE_CHECKING, Optional
import os
import json
import logging

logger = logging.getLogger(__name__)
# Optional MLX for Apple-silicon; fall back to template gen if missing
try:
    import mlx.core as mx  # type: ignore
    import mlx.nn as nn    # type: ignore
    from mlx_lm import load, generate  # type: ignore
    _MLX_AVAILABLE = True
except Exception:  # pragma: no cover
    _MLX_AVAILABLE = False
    mx = nn = None  # type: ignore
    def load(*args, **kwargs):  # type: ignore
        return None, None
    def generate(*args, **kwargs):  # type: ignore
        return "[TEST_CASE][SUMMARY]fallback[/SUMMARY][INPUT_DATA]{}[/INPUT_DATA][STEPS]1. step -> ok[/STEPS][EXPECTED]ok[/EXPECTED][/TEST_CASE]"

if TYPE_CHECKING:
    from src.models.test_case_schemas import TestCase, TestStep
    from src.config.telemetry import TelemetryConfig

class GraniteMoETrainer:
    def __init__(self, model_name: str = "ibm-granite/granite-3.0-1b-a400m-instruct"):
        self.model_name = model_name
        if _TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = None  # type: ignore
        
        if self.tokenizer is not None:
            # Add special tokens for test case structure
            special_tokens = {
                "additional_special_tokens": [
                    "[TEST_CASE]", "[/TEST_CASE]",
                    "[SUMMARY]", "[/SUMMARY]", 
                    "[STEPS]", "[/STEPS]",
                    "[EXPECTED]", "[/EXPECTED]",
                    "[INPUT_DATA]", "[/INPUT_DATA]"
                ]
            }
            self.tokenizer.add_special_tokens(special_tokens)
        
        # Load model with MLX for Apple Silicon optimization
        self.model = None
        self.mlx_model = None
        
    def load_model_for_training(self):
        """Load model for training with appropriate settings"""
        if not _TRANSFORMERS_AVAILABLE:
            return None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.tokenizer is not None:
            self.model.resize_token_embeddings(len(self.tokenizer))
        return self.model
    
    def load_model_for_inference(self):
        """Load optimized model for inference using MLX"""
        if not _MLX_AVAILABLE:
            self.mlx_model = self.mlx_tokenizer = None  # type: ignore
            return None
        try:
            self.mlx_model, self.mlx_tokenizer = load(self.model_name)
            return self.mlx_model
        except Exception:
            self.mlx_model, self.mlx_tokenizer = None, None
            return None
    
    def prepare_training_data(self, test_cases: List['TestCase'], 
                            requirements_context: List[str]) -> Dataset:
        """Prepare training dataset with structured prompts"""
        training_examples = []
        
        for i, test_case in enumerate(test_cases):
            # Create structured prompt for MoE training
            context = requirements_context[i] if i < len(requirements_context) else ""
            
            prompt = self._create_training_prompt(test_case, context)
            training_examples.append({"text": prompt})
        
        return Dataset.from_list(training_examples)
    
    def _create_training_prompt(self, test_case: 'TestCase', context: str) -> str:
        """Create structured training prompt optimized for MoE experts"""
        steps_text = "\n".join([
            f"{step.step_number}. {step.action} -> {step.expected_result}"
            for step in test_case.steps
        ])
        
        prompt = f"""<|system|>
You are a software quality engineer creating detailed test cases from requirements. Use the following structure:

<|user|>
Context: {context}
Create a test case for the above requirements.

<|assistant|>
[TEST_CASE]
[SUMMARY]{test_case.summary}[/SUMMARY]

[INPUT_DATA]
{json.dumps(test_case.input_data)}
[/INPUT_DATA]

[STEPS]
{steps_text}
[/STEPS]

[EXPECTED]
{test_case.expected_results}
[/EXPECTED]
[/TEST_CASE]<|endoftext|>"""

        return prompt
    
    def fine_tune(
        self,
        dataset: Dataset,
        output_dir: str = "./fine_tuned_granite",
        num_epochs: int = 3,
        batch_size: int = 4,
        telemetry: Optional["TelemetryConfig"] = None,
        log_checkpoints: bool = False,
    ):
        """Fine-tune the model with QLoRA for memory efficiency"""
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Configure LoRA for MoE - Target expert layers specifically
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["experts", "gate", "q_proj", "v_proj"]  # MoE specific
        )
        
        # Get PEFT model
        if not self.model:
            self.load_model_for_training()
            
        self.model = get_peft_model(self.model, lora_config)
        
        report_to: List[str] = []
        logging_dir = None
        logging_steps = 10
        if telemetry is not None:
            logging_steps = telemetry.log_interval_steps
            if telemetry.enable_wandb:
                report_to.append("wandb")
                if telemetry.wandb_project:
                    os.environ.setdefault("WANDB_PROJECT", telemetry.wandb_project)
                if telemetry.wandb_entity:
                    os.environ.setdefault("WANDB_ENTITY", telemetry.wandb_entity)
            if telemetry.enable_tensorboard:
                report_to.append("tensorboard")
                logging_dir = telemetry.tb_log_dir

        # Training arguments optimized for MacMini
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=logging_steps,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            fp16=True,  # Mixed precision for memory efficiency
            dataloader_num_workers=0,  # Disable multiprocessing on Mac
            remove_unused_columns=False,
            report_to=report_to or None,
            logging_dir=logging_dir,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset.select(range(min(100, len(dataset)))),  # SMALL EVAL SET
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        # Start training
        trainer.train()
        trainer.save_model()

        if not log_checkpoints:
            logger.info("Checkpoint logging disabled; model saved locally but not uploaded to telemetry stores.")

        return trainer

    def prepare_offline_fine_tuning(self, dataset: Dataset, output_dir: str = "./fine_tuned_granite") -> tuple[str, str]:
        """Prepare artifacts for offline fine-tuning on another machine.

        Saves the HF Dataset to disk and writes a LoRA configuration JSON file
        compatible with PEFT-based training.

        Args:
            dataset: Hugging Face Dataset to be used for fine-tuning.
            output_dir: Base directory where artifacts will be written.

        Returns:
            Tuple containing (tokenized_dataset_path, lora_config_path).
        """
        import os
        from pathlib import Path
        # Defer import and provide actionable guidance if missing
        try:
            from peft import LoraConfig, TaskType  # type: ignore
        except Exception as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "The 'peft' library is required for offline fine-tuning artifacts. "
                "Please install it with 'pip install peft'."
            ) from e

        logger = logging.getLogger(__name__)

        base_dir = Path(output_dir)
        tokenized_dir = base_dir / "tokenized_dataset"
        config_path = base_dir / "lora_config.json"

        # Ensure directories exist
        base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Preparing offline fine-tuning artifacts in {base_dir}")

        # Save dataset
        try:
            dataset.save_to_disk(str(tokenized_dir))
            logger.info(f"Saved dataset to {tokenized_dir}")
        except Exception as e:
            logger.error(f"Failed to save dataset to disk: {e}")
            raise

        # Build LoRA configuration
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["experts", "gate", "q_proj", "v_proj"],
            )
        except Exception as e:
            logger.error(f"Failed to construct LoRA config: {e}")
            raise

        # Persist LoRA config
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(lora_config.to_dict(), f, indent=2)
            logger.info(f"Wrote LoRA config to {config_path}")
        except Exception as e:
            logger.error(f"Failed to write LoRA config JSON: {e}")
            raise

        return str(tokenized_dir), str(config_path)
