from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch
from typing import List, Dict, Any, TYPE_CHECKING
import json
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate

if TYPE_CHECKING:
    from src.models.test_case_schemas import TestCase, TestStep

class GraniteMoETrainer:
    def __init__(self, model_name: str = "ibm-granite/granite-3.0-1b-a400m-instruct"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Resize token embeddings for new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        return self.model
    
    def load_model_for_inference(self):
        """Load optimized model for inference using MLX"""
        self.mlx_model, self.mlx_tokenizer = load(self.model_name)
        return self.mlx_model
    
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
    
    def fine_tune(self, dataset: Dataset, output_dir: str = "./fine_tuned_granite",
                  num_epochs: int = 3, batch_size: int = 4):
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
        
        # Training arguments optimized for MacMini
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            fp16=True,  # Mixed precision for memory efficiency
            dataloader_num_workers=0,  # Disable multiprocessing on Mac
            remove_unused_columns=False,
            report_to=None  # Disable wandb for simplicity
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
        
        return trainer
