import torch
from PIL import Image
from transformers.image_utils import load_image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
image1 = load_image(config['test_image'])

# Load processor and merged model (no PEFT needed!)
base_model = config['base_model']
model_name = base_model.split("/")[-1]
dataset_name = config['dataset_name']
dataset_short = dataset_name.split("/")[-1]  # Get just the dataset name
num_training_rows = config['num_training_rows']
merged_repo_id = f"ThomasTheMaker/{model_name}-{dataset_short}-{num_training_rows}-merged"

processor = AutoProcessor.from_pretrained(merged_repo_id)
model = Idefics3ForConditionalGeneration.from_pretrained(
    merged_repo_id,  # Load from Hugging Face Hub
    dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    device_map="auto" if DEVICE == "cuda" else None
)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": config['test_prompt']}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image1], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=config['max_new_tokens'])
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
print(generated_texts[0])