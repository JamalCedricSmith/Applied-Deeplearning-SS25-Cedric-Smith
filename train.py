import os
import json
import torch
import argparse
import decord
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer, SFTConfig

# Initialize decord for video reading
decord.bridge.set_bridge('torch')


def _load_json_array(path: str):
    """
    Load a single JSON file that contains a top-level array.

    Expected structure per item:
    [
      {"video_filename": "<file>.mp4"},
      {"question1": "...", "answer1": "..."},
      ...,
      {"questionN": "...", "answerN": "..."}
    ]
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"File {path} must contain a top-level JSON array.")
    return data


class JSONDataset(Dataset):
    """
    Dataset for cataract surgery video QA with variable-length QAs (Q1..Qn).
    ONLY supports JSON array files (no JSONL).
    """
    def __init__(self, json_file_path: str, video_directory_path: str):
        self.json_file_path = json_file_path
        self.video_directory_path = video_directory_path
        self.entries = self._load_entries()

    def _load_entries(self):
        """
        Convert each raw item to:
        {
          "video_filename": "<file>.mp4",
          "qa": [{"q": "...", "a": "..."}, ...]
        }
        """
        raw = _load_json_array(self.json_file_path)
        entries = []
        for group in raw:
            if not isinstance(group, list) or not group:
                continue
            head = group[0] if isinstance(group[0], dict) else {}
            video_filename = head.get("video_filename", "")
            qa_pairs = []
            for d in group[1:]:
                if not isinstance(d, dict):
                    continue
                for k, v in d.items():
                    if k.startswith("question"):
                        idx = k[len("question"):]
                        akey = f"answer{idx}"
                        q = v
                        a = d.get(akey, None)
                        if q is not None and a is not None:
                            qa_pairs.append({"q": q, "a": a})
            if video_filename and qa_pairs:
                entries.append({"video_filename": video_filename, "qa": qa_pairs})
        if not entries:
            raise ValueError(f"No valid entries parsed from {self.json_file_path}")
        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        video_path = os.path.join(self.video_directory_path, entry['video_filename'])
        # num_threads=1 avoids occasional reader races on some clusters
        vr = decord.VideoReader(video_path, num_threads=1)
        return vr, entry, format_data(self.video_directory_path, entry)


def format_data(video_directory_path, entry):
    """
    Build a multi-turn chat:
    - system: instruction
    - user: (video + Q1), assistant: A1
    - user: Q2, assistant: A2
    - ...
    """
    SYSTEM_MESSAGE = """You are a vision-language model specialized in analyzing cataract surgery videos.
Your task is to analyze the provided surgical video frames and extract relevant clinical information about:
- Current surgical phase/step
- Visible surgical instruments in use
- Visible anatomical structures
- Spatial relations (relative and absolute)
- Temporal context (previous/next phase, instrument changes)

Answer clearly and base your responses only on what is visible in the frames."""

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        }
    ]

    qa = entry.get("qa", [])
    video_url = os.path.join(video_directory_path, entry["video_filename"])

    if not qa:
        conversation.append({
            "role": "user",
            "content": [
                {"type": "video", "video": video_url},
                {"type": "text", "text": "Which phase of the surgery are we currently at?"}
            ],
        })
        return conversation

    # First turn: include video + first question
    conversation.append({
        "role": "user",
        "content": [
            {"type": "video", "video": video_url},
            {"type": "text", "text": qa[0]["q"]},
        ],
    })
    conversation.append({
        "role": "assistant",
        "content": [{"type": "text", "text": qa[0]["a"]}],
    })

    # Remaining QAs: text only
    for pair in qa[1:]:
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": pair["q"]}],
        })
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": pair["a"]}],
        })

    return conversation


def sample_images(vr, sample_fps=2):
    """
    Sample ~sample_fps frames per second across the clip and return
    a tensor shaped (T, C, H, W) [uint8], which matches what the
    Qwen2.5-VL fast video processor expects for normalization.
    """
    num_frames = vr._num_frame
    avg_fps = max(1, int(vr.get_avg_fps()))
    total_secs = num_frames / float(avg_fps)
    target_frames = max(1, int(total_secs * sample_fps))
    step = max(1, num_frames // target_frames)
    indices = list(range(0, num_frames, step))[:target_frames]

    # decord returns (T, H, W, C) uint8
    frames_thwc = vr.get_batch(indices)            # decord NDArrays (on CPU)
    # Convert to torch and permute to (T, C, H, W)
    frames_tc_hw = frames_thwc.to(torch.uint8).permute(0, 3, 1, 2).contiguous()
    return frames_tc_hw


def collate_fn(batch):
    vrs, _, examples = zip(*batch)

    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    # Each video: (T, C, H, W) uint8
    video_inputs = [sample_images(vr) for vr in vrs]

    # Let the processor handle resizing/normalization/padding across variable T
    model_inputs = processor(
        text=texts,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )

    # mask special tokens & pads in labels
    image_tokens = [151652, 151653, 151655, 151656]
    labels = model_inputs["input_ids"].clone()
    for i in range(labels.size(0)):
        for tok in image_tokens:
            labels[i][labels[i] == tok] = -100
        labels[i][labels[i] == processor.tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Qwen2.5-VL model on cataract surgery videos")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--hf_token", type=str)
    parser.add_argument("--train_data_path", type=str,
                        default="datasets/cataract1k/train_qa_pairs.json")
    parser.add_argument("--train_video_dir", type=str,
                        default="datasets/cataract1k/videos/train")
    parser.add_argument("--val_data_path", type=str,
                        default="datasets/cataract1k/val_qa_pairs.json")
    parser.add_argument("--val_video_dir", type=str,
                        default="datasets/cataract1k/videos/val")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    # Hub-related args (kept for optional upload, but fully optional)
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--hub_model_id", type=str, default="qwen2.5-vl-7b-instruct-cataract1k")

    parser.add_argument("--use_qlora", action="store_true", default=True)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--save_adapter", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="./qwen2.5-vl-7b-instruct-cataract1k")
    return parser.parse_args()


def main():
    global processor
    args = parse_args()

    bnb_config = None
    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.bfloat16
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config if args.use_qlora else None,
        device_map="auto",
    )

    processor = Qwen2_5_VLProcessor.from_pretrained(
        args.model_id, padding_side="right", use_fast=True
    )

    lora_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.r,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = JSONDataset(args.train_data_path, args.train_video_dir)
    val_dataset = JSONDataset(args.val_data_path, args.val_video_dir)

    # Single-GPU defaults (adjust as needed)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # ---- Hub kwargs only if pushing ----
    hub_kwargs = {}
    if args.push_to_hub:
        hub_kwargs = {
            "push_to_hub": True,
            "hub_model_id": args.hub_model_id,
            "hub_token": args.hf_token,
        }

    training_args = SFTConfig(
        output_dir=args.save_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_steps=10,
        eval_strategy="steps",            # correct key for TRL SFTConfig
        save_strategy="steps",
        save_steps=100,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        **hub_kwargs,                      # no Hub calls unless --push_to_hub
    )
    training_args.remove_unused_columns = False

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=processor.tokenizer,
    )

    trainer.train()

    if args.save_adapter:
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
