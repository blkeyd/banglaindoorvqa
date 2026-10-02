#!/usr/bin/env python3
"""
02_run_model.py -- zero-shot VQA runner for BanglaIndoorVQA (DiB baseline).

Runs one open vision-language model over items.jsonl and writes one JSON line
per answer, immediately, so nothing is lost if the session dies.

Covers all three models from the plan behind --model-path:
    google/gemma-3-12b-it            (gated: accept the licence + HF token)
    Qwen/Qwen3-VL-8B-Instruct        (open)
    OpenGVLab/InternVL3_5-8B-HF      (open -- note the -HF suffix, see below)

Fixed experimental settings from Section 5 of the plan, identical for all three:
    greedy decoding, thinking mode off, 768 px short side,
    32 new tokens for Group A, 96 for Group B, one fixed Bangla instruction.

Hardware assumptions (Section 2): NVIDIA T4, Turing.
    torch.float16      -- Turing has no bf16
    sdpa attention     -- Turing has no FlashAttention kernels
"""

import argparse
import glob as globmod
import json
import os
import sys
import time
from datetime import datetime, timezone

# ----------------------------------------------------------------------------
# The fixed prompt. Section 5 of the plan. This exact string goes in the paper.
# Do not edit it: it must be byte-identical across all three models.
# ----------------------------------------------------------------------------
BANGLA_INSTRUCTION = (
    "নিচের ছবিটি দেখে প্রশ্নের উত্তর বাংলায় সংক্ষেপে দাও। "
    "শুধু উত্তরটি লেখো, অন্য কিছু নয়।"
)

GROUP_B_TYPES = {"obj_function", "assistive"}

# Short names used in output filenames, so a merge script can tell runs apart.
KNOWN_TAGS = {
    "google/gemma-3-12b-it": "gemma3-12b",
    "google/gemma-3-4b-it": "gemma3-4b",
    "Qwen/Qwen3-VL-8B-Instruct": "qwen3vl-8b",
    "Qwen/Qwen3-VL-4B-Instruct": "qwen3vl-4b",
    "OpenGVLab/InternVL3_5-8B-HF": "internvl3_5-8b",
    "OpenGVLab/InternVL3_5-4B-HF": "internvl3_5-4b",
}

# Submodules kept OUT of 4-bit unless --quantize-vision is passed. Quantising
# the image encoder saves little VRAM and costs accuracy. Names that do not
# exist in a given model are simply ignored, so one list covers all three.
VISION_MODULES = [
    "vision_tower",          # Gemma 3
    "visual",                # Qwen3-VL
    "vision_model",          # InternVL
    "multi_modal_projector",
    "mlp1",
]


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def has_bangla(s):
    return any("\u0980" <= ch <= "\u09FF" for ch in s)


def derive_tag(model_path):
    if model_path in KNOWN_TAGS:
        return KNOWN_TAGS[model_path]
    base = os.path.basename(model_path.rstrip("/")).lower()
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in base)


def expand_paths(patterns):
    """Accept files, directories and globs. Return a de-duplicated file list."""
    out = []
    for p in patterns or []:
        if os.path.isdir(p):
            out.extend(sorted(globmod.glob(os.path.join(p, "*.jsonl"))))
        elif any(ch in p for ch in "*?["):
            out.extend(sorted(globmod.glob(p)))
        elif os.path.isfile(p):
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def read_done_ids(path):
    """Collect qa_ids already answered. Tolerates a truncated final line."""
    ids = set()
    if not os.path.isfile(path):
        return ids
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # half-written line from a killed session
            qid = rec.get("qa_id")
            if qid:
                ids.add(qid)
    return ids


# ----------------------------------------------------------------------------
# model loading
# ----------------------------------------------------------------------------

def load_model(args):
    """Imported lazily so --dry-run needs no GPU and no torch."""
    import torch
    from transformers import (
        AutoProcessor,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
    )

    if "internvl" in args.model_path.lower() and not args.model_path.lower().endswith("-hf"):
        print("\n  WARNING: InternVL ships in two formats. The plain repo is the")
        print("  GitHub format and needs model.chat() plus custom code -- it will")
        print("  probably fail to load here. Use the transformers-native one:")
        print("      --model-path OpenGVLab/InternVL3_5-8B-HF\n")

    skip = ["lm_head"] if args.quantize_vision else ["lm_head"] + VISION_MODULES

    dt = getattr(torch, args.dtype)
    cdt = getattr(torch, args.compute_dtype) if args.compute_dtype else dt
    if args.dtype != "float16":
        print(f"  NOTE: weight dtype is {args.dtype}, not the plan's float16. "
              f"Record this in the paper.")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=cdt,
        llm_int8_skip_modules=skip,
    )

    print(f"  loading processor from {args.model_path} ...", flush=True)
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )

    print("  loading weights in 4-bit (first run downloads ~17-24 GB) ...", flush=True)
    kw = dict(
        quantization_config=bnb,
        device_map={"": 0},
        attn_implementation="sdpa",             # Turing: no FlashAttention
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, dtype=dt, **kw
        )
    except TypeError:
        # transformers < 4.56 spells it torch_dtype
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, torch_dtype=dt, **kw
        )

    model.eval()

    # Greedy decoding, thinking mode off. Clearing the sampling fields stops
    # generate() warning about them being set while do_sample is False.
    gc = model.generation_config
    gc.do_sample = False
    gc.num_beams = 1
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None

    print(f"  loaded. device={model.device}, dtype=float16, attn=sdpa", flush=True)
    return processor, model


def build_inputs(processor, model, image, question, torch, dtype):
    """
    Two-step path: render the chat template to text, then let the processor
    attach the image. Works unchanged for Gemma 3, Qwen3-VL and InternVL-HF.
    Falls back to the single-step tokenising path if a template needs it.
    """
    user_text = f"{BANGLA_INSTRUCTION}\n\n{question}"
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": user_text}],
    }]

    try:
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt")
    except Exception:
        messages[0]["content"][0] = {"type": "image", "image": image}
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    # Gemma's chat template already emits <bos>, and the tokenizer adds a
    # second one. Two BOS tokens degrade the prompt, so drop the duplicate.
    bos = getattr(processor.tokenizer, "bos_token_id", None)
    ids = inputs.get("input_ids")
    if bos is not None and ids is not None and ids.shape[-1] > 1 \
            and ids[0, 0] == bos and ids[0, 1] == bos:
        for k in ("input_ids", "attention_mask", "token_type_ids"):
            if k in inputs and inputs[k] is not None:
                inputs[k] = inputs[k][:, 1:]

    # .to() with a dtype casts float tensors only, so input_ids stay integer.
    return inputs.to(device=model.device, dtype=dtype)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Zero-shot VQA runner for BanglaIndoorVQA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--items", default="items.jsonl",
                    help="shuffled item list from 01_build_items.py")
    ap.add_argument("--images", default="images_768",
                    help="resized image folder. Never point this at images/")
    ap.add_argument("--out-dir", default=".",
                    help="where the responses file is written")
    ap.add_argument("--model-path", default="google/gemma-3-12b-it",
                    help="HF repo id, or a local model directory")
    ap.add_argument("--model-tag", default=None,
                    help="short name used in the output filename")
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker's number, counting from 0")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="how many workers in total. Same on every account")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many NEW items. 20 to smoke test")
    ap.add_argument("--prev-outputs", nargs="*", default=None,
                    help="files, folders or globs to resume from")
    ap.add_argument("--print-samples", type=int, default=3,
                    help="how many Q / gold / answer triples to print")
    ap.add_argument("--progress-every", type=int, default=25,
                    help="how often to print a progress line")
    ap.add_argument("--group-a-tokens", type=int, default=32,
                    help="answer budget for the six short types. DO NOT CHANGE")
    ap.add_argument("--group-b-tokens", type=int, default=96,
                    help="answer budget for obj_function and assistive. DO NOT CHANGE")
    ap.add_argument("--max-image-px", type=int, default=768,
                    help="safety net; shrinks anything bigger that slipped through")
    ap.add_argument("--quantize-vision", action="store_true",
                    help="put the image encoder in 4-bit too. Only if you hit OOM")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"],
                    help="weight dtype. T4 has no native bf16; only change to "
                         "work around NaN logits, and record it in the paper")
    ap.add_argument("--compute-dtype", default=None,
                    choices=["float16", "bfloat16", "float32"],
                    help="bitsandbytes 4-bit compute dtype. Defaults to --dtype")
    ap.add_argument("--max-consecutive-errors", type=int, default=20,
                    help="give up after this many failures in a row")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="needed only for non-HF-format checkpoints")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the shard and resume maths, then exit. No GPU")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.num_shards < 1:
        sys.exit("ERROR: --num-shards must be at least 1")
    if not 0 <= args.shard < args.num_shards:
        sys.exit(f"ERROR: --shard must be between 0 and {args.num_shards - 1}")
    if not os.path.isfile(args.items):
        sys.exit(f"ERROR: items file not found: {args.items}")
    if not os.path.isdir(args.images):
        sys.exit(f"ERROR: images folder not found: {args.images}")

    tag = args.model_tag or derive_tag(args.model_path)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f"responses_{tag}_shard{args.shard}of{args.num_shards}.jsonl"
    )

    # ---- load and shard -----------------------------------------------------
    items = []
    with open(args.items, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    mine = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]

    # ---- resume -------------------------------------------------------------
    done = read_done_ids(out_path)
    own = len(done)

    prev_files = expand_paths(args.prev_outputs)
    for p in prev_files:
        marker = f"of{args.num_shards}."
        if "shard" in os.path.basename(p) and marker not in os.path.basename(p):
            print(f"  NOTE: {os.path.basename(p)} came from a different worker "
                  f"count than --num-shards {args.num_shards}")
        done |= read_done_ids(p)

    todo = [it for it in mine if it["qa_id"] not in done]
    if args.limit > 0:
        todo = todo[:args.limit]

    print("=" * 68)
    print(f"  model            : {args.model_path}")
    print(f"  tag              : {tag}")
    print(f"  shard            : {args.shard} of {args.num_shards}")
    print(f"  items total      : {len(items)}")
    print(f"  items this shard : {len(mine)}")
    print(f"  already done in this shard: {own}")
    if prev_files:
        print(f"  resumed from     : {len(prev_files)} earlier file(s), "
              f"{len(done) - own} extra qa_id(s)")
    print(f"  left to do       : {len(mine) - len([i for i in mine if i['qa_id'] in done])}")
    print(f"  will process now : {len(todo)}" + (f"  (--limit {args.limit})" if args.limit else ""))
    print(f"  output           : {out_path}")
    print("=" * 68, flush=True)

    if args.dry_run:
        for it in todo[:3]:
            print(f"    next: {it['qa_id']}  [{it.get('qa_type')}]  {it['image_file']}")
        print("\n  dry run only, nothing was generated.")
        return

    if not todo:
        print("  nothing left to do for this shard.")
        return

    # ---- model --------------------------------------------------------------
    import torch
    from PIL import Image

    processor, model = load_model(args)
    run_dtype = getattr(torch, args.dtype)

    # ---- generate -----------------------------------------------------------
    fout = open(out_path, "a", encoding="utf-8")
    latencies, printed, resized, errors, consecutive = [], 0, 0, 0, 0
    empty = no_bangla = long_answers = 0
    t_start = time.time()

    for n, it in enumerate(todo, 1):
        img_path = os.path.join(args.images, it["image_file"])
        budget = (args.group_b_tokens
                  if it.get("group") == "B" or it.get("qa_type") in GROUP_B_TYPES
                  else args.group_a_tokens)
        t0 = time.time()
        try:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            if min(w, h) > args.max_image_px:
                s = args.max_image_px / min(w, h)
                image = image.resize((round(w * s), round(h * s)), Image.BICUBIC)
                resized += 1

            inputs = build_inputs(processor, model, image, it["question"], torch, run_dtype)
            in_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=budget, do_sample=False)

            answer = processor.decode(out[0][in_len:], skip_special_tokens=True).strip()
            consecutive = 0

        except Exception as e:
            errors += 1
            consecutive += 1
            answer = ""
            print(f"  [error {errors}] {it['qa_id']}: {type(e).__name__}: {e}", flush=True)
            if consecutive >= args.max_consecutive_errors:
                print(f"\n  ABORTING: {consecutive} failures in a row.", flush=True)
                break

        dt = time.time() - t0
        latencies.append(dt)

        fout.write(json.dumps({
            "qa_id": it["qa_id"],
            "model_id": args.model_path,
            "raw_output": answer,
            "latency_s": round(dt, 3),
            "run_ts": utc_now(),
        }, ensure_ascii=False) + "\n")
        fout.flush()
        if n % 50 == 0:
            os.fsync(fout.fileno())

        if not answer.strip():
            empty += 1
        else:
            if not has_bangla(answer):
                no_bangla += 1
            if len(answer.split()) > 12:
                long_answers += 1

        if printed < args.print_samples:
            printed += 1
            print(f"\n  --- sample {printed} --- {it['qa_id']}  [{it.get('qa_type')}]")
            print(f"  Q    : {it['question']}")
            print(f"  gold : {it.get('gold', '')}")
            print(f"  model: {answer}")
            print(f"  ({dt:.1f}s, budget {budget} tokens)", flush=True)

        if n % args.progress_every == 0:
            rate = sum(latencies) / len(latencies)
            print(f"  [{n}/{len(todo)}]  {rate:.2f}s/item avg  "
                  f"{errors} error(s)", flush=True)

    fout.close()

    # ---- summary ------------------------------------------------------------
    n_done = len(latencies)
    if not n_done:
        print("\n  nothing completed.")
        return

    overall = sum(latencies) / n_done
    warm_slice = latencies[3:] if n_done > 4 else latencies
    warm = sum(warm_slice) / len(warm_slice)

    print("\n" + "=" * 68)
    print(f"  completed            : {n_done} items in {time.time() - t_start:.0f}s")
    print(f"  errors               : {errors}")
    print(f"  oversized images fixed: {resized}")
    print(f"  seconds per item     : {overall:.2f} overall, {warm:.2f} once warm")
    print()
    print(f"  Section 6 arithmetic -- hours for all {len(items):,} QA at this speed:")
    for w in (1, 2, 3, 4, 6, 8):
        hours = (len(items) / w) * warm / 3600
        flag = "  <-- fits one 12h session" if hours <= 12 else ""
        print(f"    {w} worker(s):{hours:8.1f} h{flag}")
    print()
    print("  Wednesday-gate sanity check on this session's answers:")
    print(f"    empty answers        : {empty}/{n_done}")
    print(f"    no Bangla characters : {no_bangla}/{n_done}")
    print(f"    longer than 12 words : {long_answers}/{n_done}")
    print()
    print("  These counters do not replace reading the answers by eye.")
    print(f"  output: {out_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
