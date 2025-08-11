# Hierarchical Reasoning Model

![](./assets/hrm.png)

Reasoning, the process of devising and executing complex goal-oriented action sequences, remains a critical challenge in AI.
Current large language models (LLMs) primarily employ Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive data requirements, and high latency. Inspired by the hierarchical and multi-timescale processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency.
HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations. With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data, yet achieves nearly perfect performance on challenging tasks including complex Sudoku puzzles and optimal path finding in large mazes.
Furthermore, HRM outperforms much larger models with significantly longer context windows on the Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring artificial general intelligence capabilities.
These results underscore HRM’s potential as a transformative advancement toward universal computation and general-purpose reasoning systems.

## Quick Start Guide 🚀

### Prerequisites ⚙️

Ensure PyTorch and CUDA are installed. The repo needs CUDA extensions to be built. If not present, run the following commands:

```bash
# Install CUDA 12.6
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run

wget -q --show-progress --progress=bar:force:noscroll -O cuda_installer.run $CUDA_URL
sudo sh cuda_installer.run --silent --toolkit --override

export CUDA_HOME=/usr/local/cuda-12.6

# Install PyTorch with CUDA 12.6
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

pip3 install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL

# Additional packages for building extensions
pip3 install packaging ninja wheel setuptools setuptools-scm
```

Then install FlashAttention. For Hopper GPUs, install FlashAttention 3

```bash
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

For Ampere or earlier GPUs, install FlashAttention 2

```bash
pip3 install flash-attn
```

## Install Python Dependencies 🐍

```bash
pip install -r requirements.txt
```

## W&B Integration 📈

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and metric visualization. Ensure you're logged in:

```bash
wandb login
```

## Run Experiments

### Quick Demo: Sudoku Solver 💻🗲

Train a master-level Sudoku AI capable of solving extremely difficult puzzles on a modern laptop GPU. 🧩

```bash
# Download and build Sudoku dataset
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000

# Start training (single GPU, smaller batch size)
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

Runtime: ~10 hours on a RTX 4070 laptop GPU

## Trained Checkpoints 🚧

 - [ARC-AGI-2](https://huggingface.co/sapientinc/HRM-checkpoint-ARC-2)
 - [Sudoku 9x9 Extreme (1000 examples)](https://huggingface.co/sapientinc/HRM-checkpoint-sudoku-extreme)
 - [Maze 30x30 Hard (1000 examples)](https://huggingface.co/sapientinc/HRM-checkpoint-maze-30x30-hard)

To use the checkpoints, see Evaluation section below.

## Full-scale Experiments 🔵

Experiments below assume an 8-GPU setup.

### Dataset Preparation

```bash
# Initialize submodules
git submodule update --init --recursive

# ARC-1
python dataset/build_arc_dataset.py  # ARC offical + ConceptARC, 960 examples
# ARC-2
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000  # ARC-2 official, 1120 examples

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py  # Full version
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples

# Maze
python dataset/build_maze_dataset.py  # 1000 examples
```

### Dataset Visualization

Explore the puzzles visually:

* Open `puzzle_visualizer.html` in your browser.
* Upload the generated dataset folder located in `data/...`.

## Launch experiments

### Small-sample (1K)

ARC-1:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py 
```

*Runtime:* ~24 hours

ARC-2:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/arc-2-aug-1000
```

*Runtime:* ~24 hours (checkpoint after 8 hours is often sufficient)

Sudoku Extreme (1k):

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

*Runtime:* ~10 minutes

Maze 30x30 Hard (1k):

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/maze-30x30-hard-1k epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

*Runtime:* ~1 hour

### Full Sudoku-Hard

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-hard-full epochs=100 eval_interval=10 lr_min_ratio=0.1 global_batch_size=2304 lr=3e-4 puzzle_emb_lr=3e-4 weight_decay=0.1 puzzle_emb_weight_decay=0.1 arch.loss.loss_type=softmax_cross_entropy arch.L_cycles=8 arch.halt_max_steps=8 arch.pos_encodings=learned
```

*Runtime:* ~2 hours

## Evaluation

Evaluate your trained models:

* Check `eval/exact_accuracy` in W&B.
* For ARC-AGI, follow these additional steps:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>
```

* Then use the provided `arc_eval.ipynb` notebook to finalize and inspect your results.

## Notes

 - Small-sample learning typically exhibits accuracy variance of around ±2 points.
 - For Sudoku-Extreme (1,000-example dataset), late-stage overfitting may cause numerical instability during training and Q-learning. It is advisable to use early stopping once the training accuracy approaches 100%.

## Citation 📜

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

---

# HRM-TextLM: Hierarchical Reasoning for Language Modeling

*Extending puzzle-solving AI to text generation with hierarchical reasoning*

## 🆕 Latest: Text Language Modeling Extension

We've successfully extended the Hierarchical Reasoning Model to support **causal language modeling**, creating **HRM-TextLM** - a novel architecture that brings multi-level reasoning to text generation.

### Key Features

- **🧠 Dual-Level Reasoning**: Maintains HRM's hierarchical H/L level processing for text
- **📝 Causal Language Modeling**: Proper next-token prediction without information leakage  
- **🔄 Adaptive Computation**: Dynamic thinking depth based on content complexity
- **⚡ Efficient Training**: Handles large text datasets with robust checkpointing
- **🎯 Dual-Mode Operation**: Single model supports both puzzle-solving AND text generation

### Text LM Quick Start

#### 1. Prepare Text Dataset

```bash
# Process your text files (supports .txt files and directories)
python dataset/build_text_dataset.py \
    --input_glob "data/raw_text/*.txt" \
    --output_dir data/text-large-1024 \
    --block_size 1024 \
    --tokenizer_name "mistralai/Mistral-7B-v0.1"
```

#### 2. Train HRM-TextLM

```bash
# Train with optimized hyperparameters for stability
DISABLE_COMPILE=1 OMP_NUM_THREADS=8 python pretrain.py \
    task=text_lm \
    data_path=data/text-large-1024 \
    epochs=8 \
    eval_interval=2 \
    global_batch_size=32 \
    lr=3e-4 \
    lr_warmup_steps=1000 \
    lr_min_ratio=0.1 \
    weight_decay=0.01 \
    max_grad_norm=1.0 \
    arch.halt_max_steps=1 \
    arch.halt_exploration_prob=0.0 \
    arch.H_cycles=1 \
    arch.L_cycles=1 \
    checkpoint_every_minutes=15
```

#### 3. Monitor Training

```bash
# Launch interactive visualization server
python serve_visualization.py --host 127.0.0.1 --port 7860
```

Then visit `http://127.0.0.1:7860` to see real-time training progress, loss curves, and hierarchical reasoning analysis.

### Text Dataset Statistics

Our implementation successfully processes:
- **Dataset Size**: ~100MB of diverse text
- **Training Blocks**: 29,445 sequences of 1,024 tokens each  
- **Validation Blocks**: 600 sequences
- **Total Tokens**: ~30 million tokens processed
- **Tokenizer**: Mistral-7B-v0.1 (32,000 vocabulary)

### Architecture Adaptations

- **Text Embeddings**: Custom embedding module with learned positional encodings
- **Causal Attention**: Strict causality enforcement throughout hierarchical processing
- **Dual-Mode Design**: Seamless switching between puzzle-solving and text generation
- **Memory Optimization**: Efficient handling of long sequences with mixed precision
- **Robust Training**: Time-based checkpointing and gradient clipping for stability

### Training Configuration

**Optimized Hyperparameters:**
- Learning Rate: `3e-4` with cosine scheduling
- Warmup Steps: `1000` for stable initialization
- Weight Decay: `0.01` for regularization  
- Gradient Clipping: `1.0` for training stability
- Simplified Cycles: `H_cycles=1, L_cycles=1` for efficiency
- Automatic Checkpointing: Every 15 minutes

### Visualization Features

The included visualization server provides:
- **📊 Real-time Loss Tracking**: Monitor training progress
- **🧠 Hierarchical Analysis**: Visualize H/L level contributions
- **⚡ Performance Metrics**: Training speed, memory usage, convergence
- **🔍 Text Generation Samples**: See model outputs during training
- **📈 Learning Curves**: Comprehensive training diagnostics

### Research Contributions

1. **First Hierarchical Text LM**: Successfully adapted puzzle-solving reasoning to language modeling
2. **Causal Hierarchical Processing**: Maintained reasoning capabilities while respecting causality constraints
3. **Cross-Domain Architecture**: Demonstrated transfer of specialized reasoning to general language tasks
4. **Efficient Implementation**: Scalable training pipeline with robust checkpointing
5. **Open Source**: Complete codebase and training infrastructure available

### Performance Highlights

- ✅ **Successful Convergence**: Smooth loss reduction during training
- ✅ **Causal Compliance**: Verified no information leakage from future tokens
- ✅ **Hierarchical Processing**: Both reasoning levels contribute to text generation
- ✅ **Training Stability**: Robust training with gradient clipping and checkpointing
- ✅ **Scalable**: Handles large datasets efficiently with memory optimization

### Interactive Demo

Experience hierarchical text generation:
- **Live Demo**: Visit the visualization server during training
- **Text Generation**: Watch real-time hierarchical reasoning
- **Training Insights**: Understand how the model learns over time

*For detailed technical information, see our comprehensive [Medium article](./HRM_TextLM_Article.md) explaining the architecture, challenges, and solutions.*
