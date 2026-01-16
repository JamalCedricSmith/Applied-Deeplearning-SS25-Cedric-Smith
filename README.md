# Surgery-SFT: Cataract Surgery Video Analysis

This repository contains the code for my Applied Deep Learning (SS 25) Course Project at Ludwig Maximilian University of Munich. The project focuses on processing cataract surgery videos, generating question-answer pairs, and fine-tuning a vision-language model (Qwen2.5-VL) to analyze surgical procedures.

## Trained Model

The following fine-tuned models are available on the Hugging Face Hub:

- [Qwen2.5-VL-7B-Instruct-Cataract1K](https://huggingface.co/kida1122/qwen2.5-vl-7b-instruct-cataract1k): Fine-tuned Qwen2.5-VL model for cataract surgery video analysis

## Data Processing Pipeline

Follow these steps in order to prepare the dataset and train the model:

### 1. Download and Install the Dataset

First, download the Cataract-1K dataset from Dropbox:
- [Download Cataract-1K Dataset (reshaped into 224x224 videos)](https://www.dropbox.com/scl/fi/5ybj7gd07hd38x1pezwdr/surgery-Cataract-1K.zip?rlkey=42wja3aptub866l487k2dhfmm&dl=0)

Extract the downloaded zip file in the base directory of this repository. The phase and segment annotations are retrieved with cloning the repository. The initial structure of the datasets should look like:

```bash
surgery-sft/
└── datasets/
    └── cataract1k/
        ├── annotations/
        │   ├── phase_annotations/
        │   └── segment_annotations/
        └── videos/
```

### 2. Connect to Cluster and Install Required Packages in a Conda Environment

On the LRZ cluster:

```bash
ssh lrz-ai
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n lmu_applied python=3.10
conda activate lmu_applied
cd surgery-sft
pip install -r requirements.txt
```

---

### 3. Cut Videos into Segments

```bash
python data_utils/01_cut_videos.py
```

This script (`01_cut_videos.py`) cuts each cataract surgery video into **5-second segments** per annotated surgical phase, also including **idle segments** between phases.  
Each segment is saved under `datasets/cataract1k/videos/` and named with its case ID, phase, and timestamps (e.g. `caseX_Incision_10.0_15.0.mp4`).

**Outputs:**
- `datasets/cataract1k/videos/*.mp4`

---

### 4. Generate Object-Level Annotations

```bash
python data_utils/02_object_generation.py
```

This script parses both **phase** and **segment-level COCO annotations** to generate structured JSON data mapping:
- case → timestamp → `{ phase, detected objects, video_filename }`

Each object entry contains segmentation, bounding box, area, and object name.  
The resulting file is used to build the graph-level dataset.

**Outputs:**
- `datasets/cataract1k/case_objects.json`

---

### 5. Generate Scene Graphs

```bash
python data_utils/03_graph_generation.py
```

This step constructs **semantic scene graphs** that capture spatial and temporal relationships between instruments and anatomical structures for every 5-second segment.  
Each graph includes:
- **Nodes:** instruments and anatomical structures with spatial coordinates (grid cell, bounding box, centroid)
- **Edges:** spatial (“left-of”, “inside”, “near”) and temporal (“entered”, “exited”) relations
- **Phase transitions and key events**

**Outputs:**
- `datasets/cataract1k/graphs.json`

---

### 6. Generate Deterministic Question–Answer Pairs

```bash
python data_utils/04_qa_generation_graph.py
```

Unlike earlier LLM-based QA generation, this script produces **deterministic and reproducible QA pairs** directly from the scene graphs.  
It automatically creates **nine QA pairs per graph**, corresponding to the four reasoning types:

| QA Type | Example |
|----------|----------|
| Recognition | “Which phase of the surgery are we currently at?” |
| Spatial | “How are the relative positions of objects A vs. B?” |
| Verification | “Which instruments entered since previous step?” |
| Temporal | “What is the next likely phase?” |

**Outputs:**
- `datasets/cataract1k/qa_pairs_graph.json`

---

### 7. Split Dataset into Train/Validation Sets (Balanced)

```bash
python data_utils/05_split_dataset.py
```

This script divides the QA pairs into **training (90%)** and **validation (10%)** sets,  
while ensuring **balanced representation across surgical cases** through duplication of underrepresented cases.

**Outputs:**
- `datasets/cataract1k/train_qa_pairs.json`  
- `datasets/cataract1k/val_qa_pairs.json`

---

### 8. Organize Videos into Train/Validation/Test Folders

```bash
python data_utils/06_organize_videos.py
```

Automatically organizes videos into train, validation, and test folders based on filenames in the split JSONs.

**Outputs:**
```
datasets/cataract1k/videos/train/
datasets/cataract1k/videos/val/
datasets/cataract1k/videos/test/
```

---

### 9. Fine-Tune the Model

```bash
python train.py
```

This script fine-tunes **Qwen2.5-VL** on the generated QA dataset with **graph-grounded supervision**.  
Key features:
- Deterministic graph-based QA parsing (JSON array, not JSONL)
- Video frame extraction using **Decord**
- **Multi-turn conversational fine-tuning** (system, user, assistant messages)
- Supports **QLoRA** for parameter-efficient training

**Arguments Example:**

```bash
python train.py   --model_id "Qwen/Qwen2.5-VL-7B-Instruct"   --train_data_path "datasets/cataract1k/train_qa_pairs.json"   --train_video_dir "datasets/cataract1k/videos/train"   --val_data_path "datasets/cataract1k/val_qa_pairs.json"   --val_video_dir "datasets/cataract1k/videos/val"   --batch_size 4   --num_epochs 2   --learning_rate 2e-5
```

**Outputs:**
- Fine-tuned adapter or full model  
- Logs and checkpoints under `qwen2.5-vl-7b-instruct-cataract1k/`

---
