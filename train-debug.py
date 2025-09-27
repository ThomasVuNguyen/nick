import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from huggingface_hub import HfApi

# -------------------------------
# 1. Settings
# -------------------------------
USE_LORA = False
USE_QLORA = True
SMOL = True
NUM_TRAINING_ROWS = 10  # Change this to 100, 1000, 10000, etc.
NUM_VALIDATION_ROWS = 10  # Number of rows for validation (picked right after training rows)

model_id = "HuggingFaceTB/SmolVLM-Base" if SMOL else "HuggingFaceM4/Idefics3-8B-Llama3"

# -------------------------------
# 2. Processor and Model
# -------------------------------
processor = AutoProcessor.from_pretrained(model_id)

if USE_QLORA or USE_LORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
    )
    lora_config.inference_mode = False
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config if USE_QLORA else None,
        _attn_implementation="flash_attention_2",
        device_map="auto",
        dtype=torch.bfloat16,  # Fixed deprecated torch_dtype
    )
    model.add_adapter(lora_config)
    model.enable_adapters()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
else:
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    ).to("cuda")

    for param in model.model.vision_model.parameters():
        param.requires_grad = False

# -------------------------------
# 3. Load CADQuery dataset
# -------------------------------
print(f"Loading ThomasTheMaker/cadquery dataset ({NUM_TRAINING_ROWS} training + {NUM_VALIDATION_ROWS} validation rows)...")
dataset = load_dataset("ThomasTheMaker/cadquery")

# Split dataset: training rows 0 to NUM_TRAINING_ROWS-1, validation rows NUM_TRAINING_ROWS to NUM_TRAINING_ROWS+NUM_VALIDATION_ROWS-1
train_ds = dataset["train"].select(range(NUM_TRAINING_ROWS))
val_ds = dataset["train"].select(range(NUM_TRAINING_ROWS, NUM_TRAINING_ROWS + NUM_VALIDATION_ROWS))

print(f"Training dataset: {len(train_ds)} rows")
print(f"Validation dataset: {len(val_ds)} rows")

# -------------------------------
# 4. Collate function
# -------------------------------
image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")]

def collate_fn(examples):
    texts, images = [], []
    for example in examples:
        image = example["images"][0]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        text = example["texts"][0]
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Answer briefly."},
                {"type": "image"},
                {"type": "text", "text": text}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": text}
            ]}
        ]
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text_prompt.strip())
        images.append([image])

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    batch["labels"] = labels
    return batch

# -------------------------------
# 5. Training
# -------------------------------
model_name = model_id.split("/")[-1]
output_dir = f"./{model_name}-cadquery-debug{NUM_TRAINING_ROWS}"
repo_id = f"ThomasTheMaker/{model_name}-cadquery-debug{NUM_TRAINING_ROWS}"

training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Conservative increase from 1 to 2
    per_device_eval_batch_size=4,   # Conservative increase from 1 to 4
    gradient_accumulation_steps=2,  # Keep at 2 (effective batch size = 2*2 = 4)
    warmup_steps=50,                # Added warmup for better convergence
    learning_rate=2e-4,             # Slightly increased learning rate
    logging_steps=5,                # Reduced logging frequency
    eval_steps=5,                   # Evaluate every 5 steps (more frequent for small datasets)
    eval_strategy="steps",
    save_strategy="no",
    bf16=True,
    output_dir=output_dir,
    hub_model_id=repo_id,
    remove_unused_columns=False,
    gradient_checkpointing=True,    # Re-enabled for memory efficiency
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=2,       # Reduced workers to save memory
    dataloader_pin_memory=True,     # Faster data transfer to GPU
    max_grad_norm=1.0,              # Gradient clipping for stability
)

model.config.use_cache = False

# Additional memory optimizations
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

# Optimize memory usage
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

# Clear cache before training
torch.cuda.empty_cache()

# Set memory fraction to be more conservative
torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available VRAM

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()

# Save and push model
if USE_QLORA or USE_LORA:
    print("Saving adapter model...")
    # Save the adapter model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Adapter model saved to {output_dir}")
    
    # Push adapter model to hub
    model.push_to_hub(repo_id)
    processor.push_to_hub(repo_id)
    print(f"Adapter model pushed to {repo_id}")
    
    print("\n" + "="*70)
    print("ADAPTER MODEL SAVED SUCCESSFULLY!")
    print("📁 Adapter model (requires PEFT):")
    print(f"   Local: {output_dir}")
    print(f"   Hub: {repo_id}")
    print("\n🔄 To create a full merged model, run:")
    print("python merge_model.py")
    print("\n🚀 Or use the adapter model with PEFT:")
    print("from peft import PeftModel")
    print("base_model = Idefics3ForConditionalGeneration.from_pretrained('HuggingFaceTB/SmolVLM-Base')")
    print(f"model = PeftModel.from_pretrained(base_model, '{output_dir}')")
    print("="*70)
else:
    # For non-LoRA models, save and push normally
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    trainer.push_to_hub()

# -------------------------------
# 6. Automatic export of logs to CSV + PNG, then upload to Hub
# -------------------------------
log_root = os.path.join(output_dir, "runs")
if os.path.exists(log_root):
    for sub in os.listdir(log_root):
        subdir = os.path.join(log_root, sub)
        if not os.path.isdir(subdir):
            continue
        event_files = [f for f in os.listdir(subdir) if f.startswith("events.out.tfevents")]
        if not event_files:
            continue
        event_path = os.path.join(subdir, event_files[0])
        print("Exporting metrics from:", event_path)

        ea = event_accumulator.EventAccumulator(event_path)
        ea.Reload()

        scalars = []
        for tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            for e in events:
                scalars.append({
                    "step": e.step,
                    "tag": tag,
                    "value": e.value
                })

        if scalars:
            df = pd.DataFrame(scalars)
            csv_path = os.path.join(output_dir, f"training_metrics_debug{NUM_TRAINING_ROWS}.csv")
            df.to_csv(csv_path, index=False)
            print("CSV saved at", csv_path)

            # Separate training and validation loss
            train_loss_df = df[df["tag"].str.contains("train/loss")]
            eval_loss_df = df[df["tag"].str.contains("eval/loss")]
            
            if not train_loss_df.empty or not eval_loss_df.empty:
                plt.figure(figsize=(12, 6))
                
                if not train_loss_df.empty:
                    plt.plot(train_loss_df["step"], train_loss_df["value"], label="Training Loss", color="blue")
                
                if not eval_loss_df.empty:
                    plt.plot(eval_loss_df["step"], eval_loss_df["value"], label="Validation Loss", color="red")
                
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.title(f"Training and Validation Loss Curves (debug{NUM_TRAINING_ROWS})")
                plt.legend()
                plt.grid(True)
                png_path = os.path.join(output_dir, f"training_validation_loss_debug{NUM_TRAINING_ROWS}.png")
                plt.savefig(png_path)
                plt.close()
                print("Plot saved at", png_path)

                # Upload both CSV and PNG to Hugging Face Hub
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=csv_path,
                    path_in_repo=f"training_metrics_debug{NUM_TRAINING_ROWS}.csv",
                    repo_id=repo_id,
                    repo_type="model"
                )
                api.upload_file(
                    path_or_fileobj=png_path,
                    path_in_repo=f"training_validation_loss_debug{NUM_TRAINING_ROWS}.png",
                    repo_id=repo_id,
                    repo_type="model"
                )
                print("CSV and PNG uploaded to Hugging Face Hub")
