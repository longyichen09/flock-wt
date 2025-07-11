import os
from dataclasses import dataclass

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from dataset import SFTDataCollator, SFTDataset
from utils.constants import model2template


@dataclass
class LoraTrainingArguments:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: int


def train_lora(
    model_id: str, context_length: int, training_args: LoraTrainingArguments
):
    assert model_id in model2template, f"model_id {model_id} not supported"
    lora_config = LoraConfig(
    r=training_args.lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 修正模块名
    lora_alpha=training_args.lora_alpha,
    lora_dropout=training_args.lora_dropout,
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens", "lm_head"]  # 新增关键层保留
)

    # Load model in 4-bit to do qLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    training_args = SFTConfig(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_steps=200,
        learning_rate=3e-4,
        bf16=True,
        logging_steps=20,
        output_dir="outputs",
        optim="adamw_torch_fused",
        remove_unused_columns=False,
        num_train_epochs=training_args.num_train_epochs,
        max_seq_length=context_length,
        gradient_checkpointing=True,  # 传递梯度检查点配置
        tf32=True,  # 启用TF32加速
        lr_scheduler_type="cosine",  # 显式指定余弦衰减
        warmup_ratio=0.1,  # 10%步数预热
        max_grad_norm=1.0,  # 新增梯度裁剪
#attn_implementation="flash_attention_2"  # 加速训练‌:ml-citation{ref="1,7" data="citationList"}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=True,
        padding_side="right"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ["HF_TOKEN"],
    )

    # Load dataset
    dataset = SFTDataset(
        file="data/demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )

    # Define trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
    )

    # Train model
    trainer.train()

    # save model
    trainer.save_model("outputs")

    # remove checkpoint folder
    os.system("rm -rf outputs/checkpoint-*")

    # upload lora weights and tokenizer
    print("Training Completed.")


if __name__ == "__main__":
    # Define training arguments for LoRA fine-tuning
    training_args = LoraTrainingArguments(
        num_train_epochs=10,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        lora_rank=64,
        lora_alpha=128,
        lora_dropout=0.05,
    )

    # Set model ID and context length
    model_id = "microsoft/Phi-3.5-mini-instruct"
    context_length = 4096

    # Start LoRA fine-tuning
    train_lora(
        model_id=model_id, context_length=context_length, training_args=training_args
    )

