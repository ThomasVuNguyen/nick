import torch
from PIL import Image
from transformers.image_utils import load_image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
image2 = load_image("https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg")

# Load processor and merged model (no PEFT needed!)
processor = AutoProcessor.from_pretrained("ThomasTheMaker/SmolVLM-Base-cadquery-debug10-merged")
model = Idefics3ForConditionalGeneration.from_pretrained(
    "ThomasTheMaker/SmolVLM-Base-cadquery-debug10-merged",  # Load from Hugging Face Hub
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
            {"type": "image"},
            {"type": "text", "text": "Can you describe the two images?"}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
print(generated_texts[0])
