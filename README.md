# FaSTA*: Fast-Slow Toolpath Agent with Subroutine Mining for Efficient Multi-turn Image Editing

> **🎉 Accepted to ICLR 2026!**

🔗 [**ArXiv Preprint**](http://arxiv.org/abs/2506.20911)

---

## 📖 Introduction

**FaSTA**\* is an advanced agent for solving complex, multi-turn image editing tasks efficiently and economically. It enhances the principles of its predecessor, **CoSTA**\*, by integrating a novel fast-slow planning strategy. **FaSTA**\* combines the high-level planning capabilities of **Large Language Models (LLMs)** with an optimal **A\* search**, but goes a step further by *learning and reusing successful toolpaths on-the-fly*. This online mining of subroutines allows the agent to solve recurring tasks rapidly, falling back to a more deliberate, slow search only when encountering novel problems, significantly reducing computational cost and time.

![Pipeline](https://github.com/advaitgupta/FaSTAR-1/raw/main/asset/schematic.png)

This repository provides:

* The official **codebase** for **FaSTA**\*.
* Scripts to **execute and optimize** toolpaths for multi-turn image editing.
* The **inductive reasoning engine** for on-the-fly subroutine mining.

---

## 📂 Dataset

**FaSTA**\* is evaluated on the benchmark dataset introduced with CoSTA\*, which contains **121 multi-turn tasks** including both image-only and text+image instructions.

**Dataset Link**: [Huggingface Dataset](https://huggingface.co/datasets/advaitgupta/CoSTAR)

---

## **Features**

✅ **Adaptive Fast-Slow Planning** – First attempts a *Fast Plan* using previously mined subroutines; only reverts to a localized A\* *Slow Path* search if the fast approach fails.  
✅ **Online Subroutine Mining** – Continuously logs execution traces and uses an LLM to perform inductive reasoning, extracting reusable, successful toolpaths and their activation rules.  
✅ **Cost-Quality Pareto Control** – A single hyperparameter, **α (alpha)**, allows users to balance the trade-off between runtime cost and final visual quality.  
✅ **Trace-Driven Verification** – Mined subroutines are only accepted if they are verified to lower cost more than they impact quality on a hold-out set of examples.  
✅ **Transparent Knowledge Base** – All learned rules are stored in a human-readable `subroutine_rule_table.json` file, which is easy to inspect, edit, or extend.  
✅ **Multimodal & Multi-turn** – Seamlessly handles complex instructions involving both image-based tasks (e.g., object recoloring) and text manipulation within images.  
✅ **Supports 24 AI Tools** – Comes ready with a wide array of tools including **YOLOv7, GroundingDINO, SAM, Stable Diffusion (in-paint, out-paint), Real-ESRGAN, EasyOCR, CLIP, GPT-4o,** and more.  

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tianyi-lab/FaSTAR.git
cd FaSTAR
```

### 2. Create Conda Environment

This command will set up the required environment from the provided file.

```bash
conda env create -f environment.yml
```

*Note: Optional large models for tools like Stable Diffusion and YOLO are downloaded automatically on their first use if `--auto_install` is enabled.*

### 3. Download Pre-trained Checkpoints

The required pre-trained model checkpoints must be downloaded from the Google Drive link provided in `checkpoints/checkpoints.txt` and extracted into the `checkpoints/` folder.

### 4. Set API Keys

Set the following environment variables. Alternatively, you can hard-code the keys in the `run.py` script.

```bash
export OPENAI_API_KEY="sk-..."
export STABILITY_API_KEY="..."
```

---

## 🚀 Usage

To execute **FaSTA**\* on an image with a prompt, run:

```bash
python run.py \
  --image path/to/image.jpg \
  --prompt "Your detailed editing instruction" \
  --output Chain.json \
  --output_image final.png \
  --alpha 0
```

**Example:**

```bash
python run.py \
  --image inputs/sample.jpg \
  --prompt "Replace the cat with a dog, then recolor the bench to pink" \
  --output Chain.json \
  --output_image final_output.png \
  --alpha 0
```

- `--image`: Path to the input image.
- `--prompt`: The instruction for the editing task.
- `--output`: Path to save the generated subtask chain.
- `--output_image`: Path to save the final edited image.
- `--alpha`: Cost-quality trade-off parameter. `0` prioritizes quality, `2` prioritizes cost.

### Interactive Demo

For a detailed, step-by-step visualization of the entire process, refer to `Demo.ipynb`. This interactive Jupyter Notebook explains the complete workflow, helps in understanding the codebase, and can be easily adapted to run your own custom tasks.

---

## 🧠 How It Works: Fast-Slow Planning

The core of FaSTA\* is its hybrid planning approach. Instead of always performing a costly search, it first tries a "fast" path using learned knowledge, and only falls back to a "slow" search when necessary.

1. **Subtask & Subroutine Generation**: The user's prompt is first broken down into a linear chain of subtasks (e.g., 1. "replace cat with dog", 2. "recolor bench to pink"). For each subtask, the agent consults its `subroutine_rule_table.json` to find potential pre-learned toolpaths (subroutines).

2. **Fast Path Attempt**: The agent selects the most promising subroutine based on its historical cost and quality, adjusted by the `alpha` parameter. It then executes this sequence of tools.

3. **Continuous Quality Check**: After each tool in the subroutine runs, the output is evaluated by a Vision Language Model (VLM). If the quality of the edit drops below a set threshold (`QUALITY_THRESHOLD`), the fast path is considered a failure.

4. **Slow Path Fallback**: If the fast path fails or if no suitable subroutine was found, the agent reverts to the slow path. This involves:
    * Building a focused **tool subgraph** containing only the tools relevant to the current, failed subtask.
    * Running an **A\* search** on this smaller graph to find the optimal, highest-quality toolpath from scratch.

5. **Trace and Learn**: Every execution, whether successful or failed, is recorded in `traces/trace_buffer.jsonl`. This data becomes the foundation for the inductive reasoning engine to learn new, better subroutines in the future.

---

## 🧠 How It Works: Inductive Reasoning

A key innovation in **FaSTA**\* is its ability to learn from experience. The workflow is as follows:

1. **Logging**: Every task execution is logged as a compact JSON entry in `traces/trace_buffer.jsonl`. This trace includes the subtask, the toolpath chosen, cost, quality, and inferred context features.

2. **Triggering**: After a set number of new traces (e.g., 20), the inductive reasoning script (`run1.py`) is triggered.

3. **Rule Proposal**: An LLM analyzes the recent traces to propose new or updated subroutines, including activation rules based on context.

4. **Validation**: Each proposed rule is tested on a small validation set. A rule is only accepted if its *net benefit* (reduction in cost versus any reduction in quality) is positive.

5. **Self-Improvement**: Validated rules are merged into the `subroutine_rule_table.json`, making the agent progressively faster and more efficient as it completes more tasks. Over time, the vast majority of subtasks can be solved using these learned "fast plans" without needing slow A\* search.

---

## 📝 Running Individual Components

*The **main functions** in the following scripts need to be **uncommented**, and the **paths, hyperparameters, and API keys** must be **modified** before execution.*

### 1. Generate a Subtask Chain

Modify `subtask_chain.py` by providing the **input image path and prompt**, then run:
```bash
python subtask_chain.py 
```

### 2. Generate a Subroutine Chain

Modify `subroutine.py` by providing the **input image path, subtask chain path and prompt**, then run:
```bash
python subroutine.py
```

### 3. Build a Tool Subgraph

Modify `tool_subgraph.py` to use a generated subtask chain, then execute:
```bash
python tool_subgraph.py 
```

### 4. Run A\* Search for Optimal Toolpath

Modify `astar_search.py` with updated paths and hyperparameters to run the slow-path search on a subgraph:
```bash
python astar_search.py 
```

### 5. Run Inductive Reasoning

To manually trigger the rule mining process on the existing trace buffer, run:
```bash
python run1.py --force
```

---

## 📁 Directory Structure

```
FaSTAR/
├── checkpoints/
│   └── checkpoints.txt
├── configs/
│   ├── tools.yaml
│   └── trace_requirements.yaml
├── dataset/
│   └── object_recoloration/
│       ├── 1.png
│       ├── 1.txt
│       └── ...
├── inputs/
│   └── sample.jpg
├── outputs/
│   ├── Chain.json
│   └── final_output.png
├── prompts/
│   └── ...
├── requirements/
│   └── ...
├── results/
│   └── ...
├── tools/
│   ├── groundingdino.py
│   ├── sam.py
│   ├── stabilityinpaint.py
│   └── ...
├── traces/
│   └── trace_buffer.jsonl
├── .gitignore
├── LICENSE
├── README.md
├── astar_search.py
├── environment.yml
├── fast_slow_planning.py
├── inductive_reasoning.py
├── run.py
├── run1.py
├── subtask_chain.py
├── subroutine.py
├── subroutine_rule_table.json
└── tool_subgraph.py
```

---

## 📊 Benchmark Results

**FaSTA**\* was evaluated on the 121-task CoSTA\* benchmark, demonstrating significant cost savings with minimal impact on quality.

| Metric (α=1) | CoSTA\* | **FaSTA**\* |
| :--- | :---: | :---: |
| Quality ↑ | **0.94** | 0.91 |
| Cost (seconds) ↓ | 58.2 s | **29.5 s** |
| **Cost Saving %** | **–** | **49%** |

---

## 📜 Citation

If you find this work useful, please consider citing the foundational paper:

```bibtex
@misc{gupta2025fastafastslowtoolpathagent,
      title={FaSTA$^*$: Fast-Slow Toolpath Agent with Subroutine Mining for Efficient Multi-turn Image Editing}, 
      author={Advait Gupta and Rishie Raj and Dang Nguyen and Tianyi Zhou},
      year={2025},
      eprint={2506.20911},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.20911}, 
}
```

