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
    lora_dropout: float
    learning_rate: float = 5e-4  # H100可以使用更大的学习率
    weight_decay: float = 0.01   # 添加权重衰减
    warmup_ratio: float = 0.05   # 预热步数比例


def train_lora(
    model_id: str, context_length: int, training_args: LoraTrainingArguments
):
    """使用 LoRA 微调模型，针对 H100 SXM 优化"""
    assert model_id in model2template, f"model_id {model_id} not supported"
    
    # 设置tokenizers并行化
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    # 设置CUDA内存分配器配置
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # 优化 LoRA 配置
    lora_config = LoraConfig(
        r=training_args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"],
        bias="none"
    )

    # H100显存充足，但仍需要合理使用
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=False,
    )

    # 设置环境变量优化性能
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["CUDA_AUTO_TUNE"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["MAX_JOBS"] = "24"

    # 启用 TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    training_args = SFTConfig(
        per_device_train_batch_size=training_args.per_device_train_batch_size,  # 减小batch size以适应内存
        gradient_accumulation_steps=training_args.gradient_accumulation_steps * 2,  # 增加梯度累积来补偿
        num_train_epochs=training_args.num_train_epochs,
        max_seq_length=context_length,
        
        # 优化器设置
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        warmup_ratio=training_args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",  # 使用8-bit Adam优化器节省内存
        
        # 混合精度训练
        bf16=True,
        tf32=True,
        
        # 性能优化
        gradient_checkpointing=True,  # 启用梯度检查点以节省内存
        logging_steps=10,
        save_strategy="epoch",
        output_dir="outputs",
        
        # 数据加载优化
        dataloader_num_workers=4,  # 减少worker数量以降低内存使用
        group_by_length=True,
        remove_unused_columns=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=True,
        padding_side="right"  # 确保padding_side为right
    )

    # 清理CUDA缓存
    torch.cuda.empty_cache()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ["HF_TOKEN"],
        load_in_8bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )

    # 使用torch.compile加速
    if torch.cuda.get_device_capability()[0] >= 8:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("Model successfully compiled")
        except Exception as e:
            print(f"Warning: Failed to compile model: {e}")

    dataset = SFTDataset(
        file="data/demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
    )

    trainer.train()
    trainer.save_model("outputs")
    os.system("rm -rf outputs/checkpoint-*")
    print("Training Completed.")


if __name__ == "__main__":
    training_args = LoraTrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=8,  # 减小batch size
        gradient_accumulation_steps=4,   # 增加梯度累积步数
        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.05,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
    )

    model_id = "microsoft/Phi-3.5-mini-instruct"
    context_length = 2048

    train_lora(
        model_id=model_id,
        context_length=context_length,
        training_args=training_args,
    )

