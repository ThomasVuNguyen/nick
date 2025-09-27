# SmolVLM Fine-tuning Project

A simple and efficient way to fine-tune SmolVLM models with LoRA/QLoRA for vision-language tasks.

## Quick Start

1. **Setup environment:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Configure training:**
Edit `config.json` to set your parameters:
- `base_model`: Model to fine-tune (e.g., "HuggingFaceTB/SmolVLM-500M-Base")
- `dataset_name`: Dataset to use (e.g., "ThomasTheMaker/cadquery")
- `num_training_rows`: Number of training examples
- `num_validation_rows`: Number of validation examples
- `learning_rate`, `batch_size`, etc.

3. **Train the model:**
```bash
python train-debug.py
```

4. **Create merged model:**
```bash
python merge_model.py
```

5. **Test inference:**
```bash
python run.py
```

## Configuration

All settings are in `config.json`:
- **Model settings**: base model, dataset
- **Training settings**: epochs, batch size, learning rate
- **Inference settings**: test image, prompt, max tokens

## Output

- **Adapter model**: `{base_model}-{dataset}-{num_rows}/` (requires PEFT)
- **Merged model**: `{base_model}-{dataset}-{num_rows}-merged/` (standalone)
- **Hugging Face Hub**: Both models uploaded automatically

**Example naming:**
- `SmolVLM-Base-cadquery-10/` (adapter)
- `SmolVLM-Base-cadquery-10-merged/` (merged)