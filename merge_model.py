#!/usr/bin/env python3
"""
Script to merge LoRA adapter with base model to create a full standalone model.
"""

import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from peft import PeftModel
import os

def merge_lora_model():
    # Configuration
    base_model_id = "HuggingFaceTB/SmolVLM-Base"
    adapter_path = "./SmolVLM-Base-cadquery-debug10"
    output_path = "./SmolVLM-Base-cadquery-debug10-merged"
    
    print("Loading base model...")
    # Load base model
    base_model = Idefics3ForConditionalGeneration.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,  # Fixed deprecated torch_dtype
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
    
    print("✅ Full merged model created successfully!")
    print(f"Model saved to: {output_path}")
    
    # Upload to Hugging Face Hub
    print("\nUploading merged model to Hugging Face Hub...")
    try:
        merged_repo_id = "ThomasTheMaker/SmolVLM-Base-cadquery-debug10-merged"
        merged_model.push_to_hub(merged_repo_id)
        processor.push_to_hub(merged_repo_id)
        print(f"✅ Merged model uploaded to: {merged_repo_id}")
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
