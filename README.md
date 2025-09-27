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
- `save_steps`: How often to save checkpoints (e.g., 1000)
- `save_total_limit`: Maximum checkpoints to keep (e.g., 3) - **Keep low to prevent VRAM issues**
- `learning_rate`, `batch_size`, etc.

3. **Train the model:**
```bash
python train-debug.py
```

**🔄 Automatic Resume**: The script automatically detects and resumes from the latest checkpoint if training was interrupted.

**⏱️ Training Time**: On RTX 4090, 20,000 training rows takes approximately 8 hours to complete.

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

## Memory Management

The training script includes automatic VRAM management:
- **Checkpoint limit**: Only 3 checkpoints kept to prevent memory accumulation
- **Cache clearing**: VRAM cache cleared every 50 steps
- **Conservative memory**: Uses 85% of available VRAM
- **Final cleanup**: Memory cleared after training

**⚠️ Important**: Keep `save_total_limit` low (≤5) to prevent out-of-memory errors!

## Checkpoint Management

### **Automatic Resume**
- Script automatically detects existing checkpoints
- Resumes from the latest checkpoint if found
- No manual intervention needed

### **Manual Control**
- **Start fresh**: Delete the output directory before training
- **Resume specific checkpoint**: Modify the script to specify a checkpoint path
- **Check available checkpoints**: Look in `{model_name}-{dataset}-{rows}/checkpoint-*`

### **Checkpoint Files**
- `checkpoint-1000/` - Checkpoint at step 1000
- `checkpoint-2000/` - Checkpoint at step 2000
- `checkpoint-3000/` - Checkpoint at step 3000

## Performance Expectations

### **Training Times (RTX 4090)**
- **1,000 rows**: ~20 minutes
- **10,000 rows**: ~3-4 hours
- **20,000 rows**: ~8 hours
- **50,000 rows**: ~20 hours

### **Memory Usage**
- **VRAM**: ~1-2GB (with QLoRA)
- **RAM**: ~1-2GB
- **Disk**: ~500MB per checkpoint

### **Scaling Factors**
- **More rows** = Linear time increase
- **Higher batch size** = Faster training, more VRAM
- **More epochs** = Linear time increase
- **QLoRA vs LoRA** = Similar performance, QLoRA uses less VRAM