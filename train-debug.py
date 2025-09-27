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
NUM_TRAINING_ROWS = 2000  # Change this to 100, 1000, 10000, etc.

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
        device_map="auto"
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
print(f"Loading ThomasTheMaker/cadquery dataset ({NUM_TRAINING_ROWS} rows)...")
dataset = load_dataset("ThomasTheMaker/cadquery")

train_ds = dataset["train"].select(range(NUM_TRAINING_ROWS))
print(train_ds)

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
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=0,
    learning_rate=1e-4,
    logging_steps=1,
    save_strategy="no",
    bf16=True,
    output_dir=output_dir,
    hub_model_id=repo_id,
    remove_unused_columns=False,
    gradient_checkpointing=True,
)

model.config.use_cache = False

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
)

trainer.train()

# Push model to your Hugging Face Hub
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

            loss_df = df[df["tag"].str.contains("loss")]
            if not loss_df.empty:
                plt.figure(figsize=(10,6))
                plt.plot(loss_df["step"], loss_df["value"], label="Loss")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.title(f"Training Loss Curve (debug{NUM_TRAINING_ROWS})")
                plt.legend()
                plt.grid(True)
                png_path = os.path.join(output_dir, f"training_loss_debug{NUM_TRAINING_ROWS}.png")
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
                    path_in_repo=f"training_loss_debug{NUM_TRAINING_ROWS}.png",
                    repo_id=repo_id,
                    repo_type="model"
                )
                print("CSV and PNG uploaded to Hugging Face Hub")
