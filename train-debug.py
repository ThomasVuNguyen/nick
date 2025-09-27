import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from huggingface_hub import HfApi

# -------------------------------
# 1. Load Configuration6
# -------------------------------
with open('config.json', 'r') as f:
    config = json.load(f)

# Simple configuration
model_id = config['base_model']
dataset_name = config['dataset_name']
NUM_TRAINING_ROWS = config['num_training_rows']
NUM_VALIDATION_ROWS = config['num_validation_rows']
USE_QLORA = config['use_qlora']
USE_LORA = False

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
        dtype=torch.bfloat16,
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
print(f"Loading {dataset_name} dataset ({NUM_TRAINING_ROWS} training + {NUM_VALIDATION_ROWS} validation rows)...")
dataset = load_dataset(dataset_name)

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
dataset_short = dataset_name.split("/")[-1]  # Get just the dataset name
output_dir = f"./{model_name}-{dataset_short}-{NUM_TRAINING_ROWS}"
repo_id = f"ThomasTheMaker/{model_name}-{dataset_short}-{NUM_TRAINING_ROWS}"

training_args = TrainingArguments(
    num_train_epochs=config['num_epochs'],
    per_device_train_batch_size=config['batch_size'],
    per_device_eval_batch_size=config['eval_batch_size'],
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    warmup_steps=config['warmup_steps'],
    learning_rate=config['learning_rate'],
    logging_steps=5,
    eval_steps=config['eval_steps'],
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=config['save_steps'],
    save_total_limit=config['save_total_limit'],
    bf16=True,
    output_dir=output_dir,
    hub_model_id=repo_id,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    max_grad_norm=1.0,
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
torch.cuda.set_per_process_memory_fraction(0.85)  # Use 85% of available VRAM (more conservative)

# Add memory management callback
from transformers import TrainerCallback

class MemoryCallback(TrainerCallback):
    def __init__(self):
        self.step_count = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        # Clear cache every 50 steps to prevent memory accumulation
        if self.step_count % 50 == 0:
            torch.cuda.empty_cache()
            print(f"🧹 Cleared VRAM cache at step {self.step_count}")

memory_callback = MemoryCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    callbacks=[memory_callback],
)

# Check for existing checkpoints and resume if found
import glob
checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
if checkpoint_dirs:
    # Sort by step number to get the latest checkpoint
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))
    print(f"🔄 Found existing checkpoint: {latest_checkpoint}")
    print("📈 Resuming training from checkpoint...")
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    print("🚀 Starting fresh training...")
    trainer.train()

# Clear memory after training
torch.cuda.empty_cache()
print("🧹 Final VRAM cleanup completed")

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
    
    # Generate README for adapter model
    readme_content = f"""# {model_name}-{dataset_short}-{NUM_TRAINING_ROWS}

This is a LoRA adapter model fine-tuned on the {dataset_name} dataset.

## Model Details

- **Base Model**: {model_id}
- **Dataset**: {dataset_name}
- **Training Rows**: {NUM_TRAINING_ROWS}
- **Validation Rows**: {NUM_VALIDATION_ROWS}
- **Fine-tuning Method**: {'QLoRA' if USE_QLORA else 'LoRA'}

## Training Configuration

The model was trained with the following configuration from `config.json`:

```json
{json.dumps(config, indent=2)}
```

## Usage

### Option 1: Use with PEFT (Recommended for development)

```python
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from peft import PeftModel

# Load base model
base_model = Idefics3ForConditionalGeneration.from_pretrained("{model_id}")
processor = AutoProcessor.from_pretrained("{model_id}")

# Load adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
```

### Option 2: Create merged model

```bash
python merge_model.py
```

This will create a standalone merged model that doesn't require PEFT.

## Files

- `adapter_config.json` - LoRA adapter configuration
- `adapter_model.safetensors` - LoRA adapter weights
- `config.json` - Complete training configuration used
- `checkpoint-{step}/` - Training checkpoints (saved every {config['save_steps']} steps)
- `training_metrics_{NUM_TRAINING_ROWS}.csv` - Training metrics
- `training_validation_loss_{NUM_TRAINING_ROWS}.png` - Loss curves

## Checkpoints

Training checkpoints are saved every {config['save_steps']} steps, with a maximum of {config['save_total_limit']} checkpoints kept. You can resume training from any checkpoint:

```python
from transformers import Trainer
trainer = Trainer.from_pretrained("./checkpoint-{step}")
trainer.train(resume_from_checkpoint=True)
```

## Performance

Check the training metrics CSV and loss curves PNG for detailed performance information.
"""

    # Save README
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    # Save config.json for reference
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Upload README to Hub
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model"
        )
        print("✅ README uploaded to Hugging Face Hub")
    except Exception as e:
        print(f"❌ Failed to upload README: {e}")

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
