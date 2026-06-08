"""Knowledge distillation pipeline."""
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DPOTrainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from typing import List, Dict
import torch

class DistillationPipeline:
    def __init__(self, student_model: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.student_model_name = student_model
        self.tokenizer = AutoTokenizer.from_pretrained(student_model)
        self.model = AutoModelForCausalLM.from_pretrained(student_model, torch_dtype=torch.bfloat16, device_map="auto")
        lora_config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM")
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(self, examples: List[Dict]) -> Dataset:
        """Format examples for SFT training."""
        formatted = []
        for ex in examples:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["input"]}, {"role": "assistant", "content": ex["output"]}],
                tokenize=False, add_generation_prompt=False)
            formatted.append({"text": text})
        return Dataset.from_list(formatted)

    def train(self, train_examples: List[Dict], output_dir: str = "distilled_model"):
        dataset = self.prepare_dataset(train_examples)
        args = TrainingArguments(output_dir=output_dir, num_train_epochs=3, per_device_train_batch_size=4,
            gradient_accumulation_steps=4, learning_rate=2e-4, bf16=True, logging_steps=10,
            save_strategy="epoch", warmup_ratio=0.05)
        trainer = SFTTrainer(model=self.model, args=args, train_dataset=dataset,
            tokenizer=self.tokenizer, dataset_text_field="text", max_seq_length=2048)
        trainer.train()
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")
