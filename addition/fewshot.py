"""
Option A — Few-shot Analysis on New Models
==========================================
Runs k-shot evaluation (k in {0,1,2,5}) across CM-MMLU language pairs
using the OpenAI API, then aggregates and plots results matching Figure 4
of the CodeMixBench paper.

Requirements:
    pip install openai datasets pandas matplotlib scikit-learn sacrebleu tqdm

Usage:
    python option_a_kshot_analysis.py \
        --model gpt-3.5-turbo \
        --api_key YOUR_KEY \
        --output_dir result/ \
        --max_samples 100
"""

import os
import re
import json
import time
import argparse
import asyncio
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — mirror the paper's language pair groupings (Figure 4)
# ---------------------------------------------------------------------------
LANG_PAIRS = [
    "zh-en", "hi-en", "bn-en", "mr-en", "ne-en",
    "es-en", "fr-en", "ar-en", "ta-en", "nl-en", "de-en",
]

FAMILY_GROUPS = {
    "Romance & Germanic": ["es-en", "fr-en", "de-en", "nl-en"],
    "Sino-Tibetan":       ["zh-en"],
    "Afro-Asiatic":       ["ar-en"],
    "Indo-Aryan":         ["hi-en", "bn-en", "mr-en", "ne-en"],
    "Dravidian":          ["ta-en"],
}

K_SHOTS = [0, 1, 2, 5]

# HuggingFace dataset config names — adjust if the repo uses different split names
HF_DATASET = "CodeMixBench/CodeMixBench"
HF_CONFIG_TEMPLATE = "cm_mmlu_{lang}"   # e.g. cm_mmlu_zh_en

# ---------------------------------------------------------------------------
# Prompt templates (Appendix G.4 of the paper)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a system possessing knowledge in all subjects. "
    "You are skilled at selecting the correct answer based on multiple-choice questions. "
    "Do not include explanations in your answer."
)

def build_few_shot_prompt(examples: list[dict], target: dict) -> str:
    """
    Build a k-shot prompt. Each example/target dict must have keys:
      'question' (str)  — the full MC question with options
      'answer'   (str)  — the correct option letter, e.g. 'A'
    """
    messages = []
    for ex in examples:
        messages.append({"role": "user",   "content": f"Question: {ex['question']}"})
        messages.append({"role": "assistant", "content": f"Answer: {ex['answer']}"})
    messages.append({"role": "user", "content": f"Question: {target['question']}"})
    return messages


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_cm_mmlu(lang_pair: str, split: str = "test"):
    """
    Load CM-MMLU for a language pair from HuggingFace.
    Falls back to a synthetic stub if the dataset is unavailable
    (so the script structure can be tested offline).
    """
    config = HF_CONFIG_TEMPLATE.format(lang=lang_pair.replace("-", "_"))
    try:
        ds = load_dataset(HF_DATASET, config, split=split, trust_remote_code=True)
        log.info(f"Loaded {len(ds)} samples for {lang_pair} from HuggingFace.")
        return ds
    except Exception as e:
        log.warning(f"Could not load {lang_pair} from HuggingFace ({e}). Using stub data.")
        # Minimal stub so the rest of the pipeline runs
        stub = [
            {"question": f"Stub question {i} ({lang_pair})?\n(A) A\n(B) B\n(C) C\n(D) D",
             "answer": "A"}
            for i in range(20)
        ]
        return stub


def load_en_only(split: str = "test", n: int = 1000):
    """Load the English-only CM-MMLU baseline (randomly sampled from original MMLU)."""
    try:
        ds = load_dataset(HF_DATASET, "cm_mmlu_en_only", split=split, trust_remote_code=True)
        log.info(f"Loaded {len(ds)} en-only samples.")
        return ds
    except Exception as e:
        log.warning(f"Could not load en-only split ({e}). Using stub.")
        return [
            {"question": f"English stub question {i}?\n(A) A\n(B) B\n(C) C\n(D) D",
             "answer": "A"}
            for i in range(n)
        ]


# ---------------------------------------------------------------------------
# OpenAI async inference
# ---------------------------------------------------------------------------
async def call_openai(
    client: AsyncOpenAI,
    messages: list[dict],
    model: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_retries: int = 5,
) -> str:
    """Single async call to OpenAI chat completions with exponential backoff."""
    system = [{"role": "system", "content": SYSTEM_PROMPT}]
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=system + messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=16,   # answer is a single letter
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"OpenAI error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s.")
            await asyncio.sleep(wait)
    return ""


async def run_batch(
    client: AsyncOpenAI,
    model: str,
    items: list[dict],
    k: int,
    concurrency: int = 10,
) -> list[dict]:
    """
    Run inference over all items with k-shot prompting.
    items: list of dicts with 'question', 'answer', and optionally 'few_shot_pool'
    Returns list of dicts with original fields plus 'prediction'.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def process(item, idx):
        async with semaphore:
            # Sample k examples from the pool (excluding the current item)
            pool = item.get("few_shot_pool", [])
            examples = pool[:k]   # pool is pre-shuffled; just take first k
            messages = build_few_shot_prompt(examples, item)
            pred = await call_openai(client, messages, model)
            return {**item, "index": idx, "prediction": pred, "k": k}

    tasks = [process(item, i) for i, item in enumerate(items)]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                     desc=f"k={k}", leave=False):
        results.append(await coro)

    results.sort(key=lambda x: x["index"])
    return results


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------
def parse_answer(raw: str) -> str:
    """Extract option letter (A/B/C/D) from model output."""
    if not raw:
        return ""
    # Match 'Answer: A', 'A)', '(A)', or bare 'A'
    m = re.search(r"\b([ABCD])\b", raw.upper())
    return m.group(1) if m else raw.strip().upper()[:1]


def score_predictions(results: list[dict]) -> float:
    """Return accuracy as a float 0–100."""
    if not results:
        return 0.0
    correct = sum(
        1 for r in results
        if parse_answer(r["prediction"]) == str(r["answer"]).strip().upper()
    )
    return 100.0 * correct / len(results)


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------
def save_results(results: list[dict], path: str):
    pd.DataFrame(results).to_csv(path, index=False)
    log.info(f"Saved {len(results)} rows to {path}")


def load_results(path: str) -> list[dict]:
    return pd.read_csv(path).to_dict("records")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
async def evaluate(args):
    client = AsyncOpenAI(api_key=args.api_key)
    os.makedirs(args.output_dir, exist_ok=True)

    # Accumulate accuracy table: {lang_pair: {k: accuracy}}
    acc_table = defaultdict(dict)

    # Evaluate en-only baseline at each k
    en_data = list(load_en_only())[:args.max_samples]
    # Build few-shot pool from the same en-only data (leave-one-out style)
    for i, item in enumerate(en_data):
        item["few_shot_pool"] = [x for j, x in enumerate(en_data) if j != i][:10]

    for k in K_SHOTS:
        tag = f"{args.model.replace('/', '_')}_en_only_k{k}"
        out_path = os.path.join(args.output_dir, f"{tag}_pred.csv")

        if os.path.exists(out_path) and not args.force_rerun:
            log.info(f"[RESUME] Loading existing: {out_path}")
            res = load_results(out_path)
        else:
            res = await run_batch(client, args.model, en_data, k=k,
                                  concurrency=args.concurrency)
            save_results(res, out_path)

        acc = score_predictions(res)
        acc_table["en-only"][k] = acc
        log.info(f"en-only | k={k} | accuracy={acc:.2f}%")

    # Evaluate each language pair
    for lang in LANG_PAIRS:
        log.info(f"\n{'='*50}\nLanguage pair: {lang}\n{'='*50}")
        raw_data = list(load_cm_mmlu(lang))[:args.max_samples]

        # Build few-shot pool per item
        for i, item in enumerate(raw_data):
            item["few_shot_pool"] = [x for j, x in enumerate(raw_data) if j != i][:10]

        for k in K_SHOTS:
            tag = f"{args.model.replace('/', '_')}_{lang.replace('-','_')}_k{k}"
            out_path = os.path.join(args.output_dir, f"{tag}_pred.csv")

            if os.path.exists(out_path) and not args.force_rerun:
                log.info(f"[RESUME] Loading existing: {out_path}")
                res = load_results(out_path)
            else:
                res = await run_batch(client, args.model, raw_data, k=k,
                                      concurrency=args.concurrency)
                save_results(res, out_path)

            acc = score_predictions(res)
            acc_table[lang][k] = acc
            log.info(f"{lang} | k={k} | accuracy={acc:.2f}%")

    return acc_table


# ---------------------------------------------------------------------------
# Aggregation by language family (matches Figure 4 groupings)
# ---------------------------------------------------------------------------
def aggregate_by_family(acc_table: dict) -> dict:
    """
    Returns {family_name: {k: mean_accuracy}} averaged over member lang pairs.
    Also includes 'en-only' as a reference series.
    """
    family_acc = {}
    for family, langs in FAMILY_GROUPS.items():
        family_acc[family] = {}
        for k in K_SHOTS:
            vals = [acc_table[lp][k] for lp in langs if lp in acc_table and k in acc_table[lp]]
            family_acc[family][k] = sum(vals) / len(vals) if vals else float("nan")
    family_acc["en-only"] = acc_table.get("en-only", {})
    return family_acc


# ---------------------------------------------------------------------------
# Plotting (reproduces Figure 4 layout)
# ---------------------------------------------------------------------------
def plot_kshot_figure(family_acc: dict, model_name: str, output_path: str):
    families = list(FAMILY_GROUPS.keys())
    n_families = len(families)

    fig = plt.figure(figsize=(4 * (n_families + 1), 4))
    gs = gridspec.GridSpec(1, n_families + 1, figure=fig, wspace=0.35)

    en_only_vals = [family_acc["en-only"].get(k, float("nan")) for k in K_SHOTS]

    for col, family in enumerate(families):
        ax = fig.add_subplot(gs[0, col])
        vals = [family_acc[family].get(k, float("nan")) for k in K_SHOTS]

        ax.plot(K_SHOTS, en_only_vals, linestyle="--", color="gray",
                linewidth=1.2, label="en only")
        ax.plot(K_SHOTS, vals, marker="o", linewidth=2,
                label=model_name, color="steelblue")

        ax.set_title(family, fontsize=9, fontweight="bold")
        ax.set_xlabel("k-shot", fontsize=8)
        if col == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=8)
        ax.set_xticks(K_SHOTS)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Summary subplot: all families on one axis
    ax_all = fig.add_subplot(gs[0, n_families])
    colors = plt.cm.tab10.colors
    for i, family in enumerate(families):
        vals = [family_acc[family].get(k, float("nan")) for k in K_SHOTS]
        ax_all.plot(K_SHOTS, vals, marker="o", linewidth=1.5,
                    label=family[:12], color=colors[i])
    ax_all.plot(K_SHOTS, en_only_vals, linestyle="--", color="gray",
                linewidth=1.2, label="en only")
    ax_all.set_title("All Families", fontsize=9, fontweight="bold")
    ax_all.set_xlabel("k-shot", fontsize=8)
    ax_all.set_xticks(K_SHOTS)
    ax_all.tick_params(labelsize=7)
    ax_all.legend(fontsize=5, loc="lower right")
    ax_all.set_ylim(0, 100)
    ax_all.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(f"CM-MMLU K-shot Analysis — {model_name}", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info(f"Figure saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Save summary table
# ---------------------------------------------------------------------------
def save_summary_table(acc_table: dict, model_name: str, output_dir: str):
    rows = []
    for lang, k_acc in acc_table.items():
        for k, acc in k_acc.items():
            rows.append({"lang_pair": lang, "k_shot": k, "accuracy": round(acc, 2),
                         "model": model_name})
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_kshot_summary.csv")
    df.to_csv(path, index=False)
    log.info(f"Summary table saved to {path}")

    # Also print a pivot table for easy copy-paste into the paper
    pivot = df.pivot_table(index="lang_pair", columns="k_shot", values="accuracy")
    print("\n=== K-shot Accuracy Table ===")
    print(pivot.to_string())
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Option A: K-shot analysis on new models")
    p.add_argument("--model", default="gpt-3.5-turbo",
                   help="OpenAI model name, e.g. gpt-3.5-turbo or gpt-4o")
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""),
                   help="OpenAI API key (or set OPENAI_API_KEY env var)")
    p.add_argument("--output_dir", default="result/option_a",
                   help="Directory for prediction CSVs and figures")
    p.add_argument("--max_samples", type=int, default=200,
                   help="Max samples per language pair (use small value to limit cost)")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Number of concurrent OpenAI requests")
    p.add_argument("--force_rerun", action="store_true",
                   help="Ignore cached result files and rerun everything")
    p.add_argument("--lang_pairs", nargs="+", default=LANG_PAIRS,
                   help="Subset of language pairs to run (default: all)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    args = parse_args()

    if not args.api_key:
        raise ValueError("Provide --api_key or set OPENAI_API_KEY environment variable.")

    log.info(f"Model: {args.model}")
    log.info(f"Max samples per lang pair: {args.max_samples}")
    log.info(f"K-shots: {K_SHOTS}")
    log.info(f"Language pairs: {args.lang_pairs}")

    # Override global LANG_PAIRS if user provided a subset
    global LANG_PAIRS
    LANG_PAIRS = args.lang_pairs

    acc_table = await evaluate(args)

    family_acc = aggregate_by_family(acc_table)

    fig_path = os.path.join(args.output_dir,
                            f"{args.model.replace('/', '_')}_kshot_figure4.png")
    plot_kshot_figure(family_acc, model_name=args.model, output_path=fig_path)

    save_summary_table(acc_table, model_name=args.model, output_dir=args.output_dir)

    log.info("\nDone. Files written to: " + args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())