import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

# -------------------------------
# 1. Settings
# -------------------------------
USE_LORA = False
USE_QLORA = True
SMOL = True

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
    print(model.get_nb_trainable_parameters())
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
print("Loading ThomasTheMaker/cadquery dataset...")
dataset = load_dataset("ThomasTheMaker/cadquery")

split_ds = dataset["train"].train_test_split(test_size=0.1)
train_ds = split_ds["train"]

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

training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=1,
    optim="paged_adamw_8bit",
    bf16=True,
    output_dir=f"./{model_name}-cadquery",
    hub_model_id=f"{model_name}-cadquery",
    report_to="tensorboard",
    remove_unused_columns=False,
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
)

trainer.train()

trainer.push_to_hub()
