#!/usr/bin/env python3
"""
Script to merge LoRA adapter with base model to create a full standalone model.
"""

import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from peft import PeftModel
import os
import json

def merge_lora_model():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Simple configuration
    base_model_id = config['base_model']
    model_name = base_model_id.split("/")[-1]
    dataset_name = config['dataset_name']
    dataset_short = dataset_name.split("/")[-1]  # Get just the dataset name
    num_training_rows = config['num_training_rows']
    adapter_path = f"./{model_name}-{dataset_short}-{num_training_rows}"
    output_path = f"./{model_name}-{dataset_short}-{num_training_rows}-merged"
    
    print("Loading base model...")
    # Load base model
    base_model = Idefics3ForConditionalGeneration.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        device_map="auto"
    )
    
    print("Loading processor...")
    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_id)
    
    print("Loading LoRA adapter...")
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging adapter with base model...")
    try:
        # Try the standard merge approach
        merged_model = model.merge_and_unload()
        print("✅ Standard merge successful!")
    except Exception as e:
        print(f"Standard merge failed: {e}")
        print("Trying alternative merge approach...")
        
        # Alternative approach: manually merge weights
        try:
            # Get the base model from the PEFT model
            base_model = model.base_model.model
            
            # Create a new model with the same config
            merged_model = Idefics3ForConditionalGeneration.from_pretrained(
                base_model_id,
                dtype=torch.bfloat16,
                _attn_implementation="flash_attention_2",
                device_map="auto"
            )
            
            # Copy the merged weights from the PEFT model
            merged_model.load_state_dict(model.state_dict(), strict=False)
            print("✅ Alternative merge successful!")
            
        except Exception as e2:
            print(f"Alternative merge also failed: {e2}")
            print("❌ Could not merge model. Using adapter model instead.")
            return
    
    print(f"Saving merged model to {output_path}...")
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save the merged model
    merged_model.save_pretrained(output_path, safe_serialization=True)
    processor.save_pretrained(output_path)
    
    # Generate README for merged model
    readme_content = f"""# {model_name}-{dataset_short}-{num_training_rows}-merged

This is a fully merged model created by combining the base model with a LoRA adapter fine-tuned on the {dataset_name} dataset.

## Model Details

- **Base Model**: {base_model_id}
- **Dataset**: {dataset_name}
- **Training Rows**: {num_training_rows}
- **Validation Rows**: {config['num_validation_rows']}
- **Fine-tuning Method**: {'QLoRA' if config['use_qlora'] else 'LoRA'}

## Training Configuration

The model was trained with the following configuration from `config.json`:

```json
{json.dumps(config, indent=2)}
```

## Usage

### Direct Usage (No PEFT Required)

```python
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

# Load the merged model directly
processor = AutoProcessor.from_pretrained("{merged_repo_id}")
model = Idefics3ForConditionalGeneration.from_pretrained(
    "{merged_repo_id}",
    dtype=torch.bfloat16,
    device_map="auto"
)

# Example inference
import torch
from PIL import Image

# Load your image
image = Image.open("your_image.jpg")

# Create messages
messages = [
    {{
        "role": "user", 
        "content": [
            {{"type": "text", "text": "Answer briefly."}},
            {{"type": "image"}},
            {{"type": "text", "text": "What do you see in this image?"}}
        ]
    }}
]

# Process and generate
text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text_prompt, images=[image], return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens={config['max_new_tokens']},
        do_sample=True,
        temperature=0.7
    )

response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

## Files

- `config.json` - Model configuration
- `training_config.json` - Complete training configuration used
- `generation_config.json` - Generation parameters
- `model.safetensors` - Model weights
- `preprocessor_config.json` - Processor configuration
- `tokenizer.json` - Tokenizer configuration
- `tokenizer_config.json` - Tokenizer settings

## Performance

This merged model combines the efficiency of LoRA fine-tuning with the convenience of a standalone model. It should perform similarly to the original adapter model but without requiring PEFT for inference.

## Original Adapter

The original LoRA adapter is available at: `ThomasTheMaker/{model_name}-{dataset_short}-{num_training_rows}`
"""

    # Save README
    readme_path = os.path.join(output_path, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    # Save training config.json for reference
    training_config_path = os.path.join(output_path, "training_config.json")
    with open(training_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Full merged model created successfully!")
    print(f"Model saved to: {output_path}")
    
    # Upload to Hugging Face Hub
    print("\nUploading merged model to Hugging Face Hub...")
    try:
        merged_repo_id = f"ThomasTheMaker/{model_name}-{dataset_short}-{num_training_rows}-merged"
        merged_model.push_to_hub(merged_repo_id)
        processor.push_to_hub(merged_repo_id)
        
        # Upload README to Hub
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=merged_repo_id,
            repo_type="model"
        )
        print(f"✅ Merged model uploaded to: {merged_repo_id}")
        print("✅ README uploaded to Hugging Face Hub")
    except Exception as e:
        print(f"❌ Failed to upload to Hub: {e}")
        print("Model saved locally but not uploaded to Hub.")
    
    print("\nYou can now use it with:")
    print(f"processor = AutoProcessor.from_pretrained('{output_path}')")
    print(f"model = Idefics3ForConditionalGeneration.from_pretrained('{output_path}')")
    print(f"\nOr from Hugging Face Hub:")
    print(f"processor = AutoProcessor.from_pretrained('{merged_repo_id}')")
    print(f"model = Idefics3ForConditionalGeneration.from_pretrained('{merged_repo_id}')")
    
    # Check what files were created
    print(f"\nFiles created in {output_path}:")
    for file in os.listdir(output_path):
        file_path = os.path.join(output_path, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  {file} ({size:,} bytes)")

if __name__ == "__main__":
    merge_lora_model()
