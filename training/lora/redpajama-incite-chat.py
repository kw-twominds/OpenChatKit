import os
import json
import fire
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import transformers
import torch.nn as nn
import bitsandbytes as bnb
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# this script should take around 14GB VRAM
def main(
    model_name: str = "redpajama-incite-chat-3b-lowrank",
    dataset_path: str = "data/OIG-chip2/unified_chip2.jsonl",
    base_model: str = "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4, 
    num_epochs: int = 3,
    learning_rate: float = 3e-4, 
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: list = ["query_key_value", "xxx"],
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
):
    print(
        f"\n================================================================================\n"
        f"Training RedPajama model with params:\n"
        f"================================================================================\n"
        f"model_name:           {model_name}\n"
        f"base_model:           {base_model}\n"
        f"dataset_path:         {dataset_path}\n"
        f"batch_size:           {batch_size}\n"
        f"micro_batch_size:     {micro_batch_size}\n"
        f"num_epochs:           {num_epochs}\n"
        f"learning_rate:        {learning_rate}\n"
        f"lora_r:               {lora_r}\n"
        f"lora_alpha:           {lora_alpha}\n"
        f"lora_dropout:         {lora_dropout}\n"
        f"lora_target_modules:  {lora_target_modules}\n"
        f"wandb_project:        {wandb_project}\n"
        f"wandb_run_name:       {wandb_run_name}\n"
        f"wandb_watch:          {wandb_watch}\n"
        f"wandb_log_model:      {wandb_log_model}\n"
        f"================================================================================\n"
    )
    gradient_accumulation_steps = batch_size // micro_batch_size
    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # read datasets
    with open(dataset_path, 'r') as fp:
        data = [json.loads(x) for x in fp.readlines()]

    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    for param in model.parameters():
      param.requires_grad = False  # freeze the model - train adapters later
      if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    ## Training

    data = Dataset.from_list(data)
    data = data.map(lambda samples: tokenizer(samples['text']), batched=True)

    trainer = transformers.Trainer(
        model=model, 
        train_dataset=data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size, 
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            max_steps=200, 
            learning_rate=learning_rate, 
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            output_dir='outputs',
            report_to="wandb" if use_wandb else "none",
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    # save the trained adapter to disk
    model.save_pretrained(f"outputs/{model_name}")

if __name__ == "__main__":
    fire.Fire(main)
