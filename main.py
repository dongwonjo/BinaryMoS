import sys
sys.path.append(".")
import argparse
import os
import time
import json
from pathlib import Path
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

from binarization.binary_util import get_blocks, replace_modules
from utils.datautils import get_qat_dataset
from utils.utils import print_trainable_parameters, prepare_model_for_training
from utils.kd_utils import KDTrainer

def main(args):
    set_seed(args.seed)

    # Save Directory
    if args.save_dir:
        tm = time.localtime(time.time())
        f_name = f'{tm.tm_year}_{tm.tm_mon}_{tm.tm_mday}_{tm.tm_hour}_{tm.tm_min}_{tm.tm_sec}'
        save_dir = os.path.join(os.path.join(args.save_dir, args.model_id), f_name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, device_map='auto', use_fast=False, trust_remote_code=True)
    
    # Load Model
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map='auto', torch_dtype=torch.float16)
    model.config.use_cache = False
    prepare_model_for_training(model)
        
    print(f'Model GPU Status: {model.hf_device_map}')
    for i in range(torch.cuda.device_count()):
        print(f"Max memory_allocated (device:{i}): {torch.cuda.max_memory_allocated(device=f'cuda:{i}') / 1024**3:.2f} GiB")

    # Replace nn.Linear with BinaryMoSLinear
    replace_modules(get_blocks(model), num_expert=args.num_expert, do_train=True, print_layers=True)
    print_trainable_parameters(model)
    
    # Load dataset
    print(f"Prepare training data ({args.dataset})")
    datasets, data_collator = get_qat_dataset(args.dataset, tokenizer, args.cache_dir)
    
    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        logging_steps=1,
        save_steps=10000,
        output_dir=save_dir,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,   
        )
    
    # Prepare KDTrainer
    print(f"Loading Teacher Model")
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map='auto', torch_dtype=torch.float16)
    teacher_model.config.use_cache = False
    print(f'Teacher Model GPU Status: {teacher_model.hf_device_map}')
    for i in range(torch.cuda.device_count()):
        print(f"Max memory_allocated (device:{i}): {torch.cuda.max_memory_allocated(device=f'cuda:{i}') / 1024**3:.2f} GiB")

    trainer = KDTrainer(
        model=model,
        teacher_model=teacher_model,
        l2l_loss_scale=args.l2l_loss_scale,
        tokenizer=tokenizer,
        train_dataset=datasets,
        args=training_args,
        data_collator=data_collator,
        )

    # Train the model
    trainer.train()

    # Save model
    model.eval()
    model.config.num_expert = args.num_expert
    model.config.use_cache = True
    for param in model.parameters():
        param.data = param.data.to(torch.float16)
    model.save_pretrained(save_directory=save_dir, is_main_process=True)
    tokenizer.save_pretrained(save_directory=save_dir)
    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id", type=str, default="meta-llama/Llama-2-7b-hf"
    )
    parser.add_argument(
        "--dataset", type=str, default="c4_wiki", choices=['c4', 'wikitext2', 'c4_wiki']
    )
    parser.add_argument(
        "--save_dir", type=str, default='outputs'
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed"
    )
    parser.add_argument(
        "--num_train_epochs", type=float, default=3
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=4
    )   
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.03
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9
    )  
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999
    )  
    parser.add_argument(
        "--num_expert", type=int, default=4,
    )
    parser.add_argument(
        "--l2l_loss_scale", type=float, default=10.0,
    )

    args = parser.parse_args()

    main(args)
