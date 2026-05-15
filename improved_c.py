"""
Option C — M-index / I-index as Performance Covariate
======================================================

Joins per-instance mixing intensity (M-index, I-index) with model prediction
correctness, bins by mixing quartile, and produces:
  1. Binned accuracy vs. M-index (Figure C1)
  2. Binned accuracy vs. I-index (Figure C2)
  3. Interaction plot: mixing intensity × language family (Figure C3)
  4. Logistic regression: M-index, I-index, cross_family → P(correct)

Important methodological update:
  - The original fallback heuristic treated all ASCII tokens as English.
  - That is bad for same-script pairs such as es-en, fr-en, de-en, nl-en.
  - This version adds a better Spanish-English token heuristic so es-en does
    not label every Latin/ASCII Spanish word as English.

Requirements:
    pip install openai datasets pandas matplotlib seaborn scipy scikit-learn statsmodels tqdm

Usage:
    # Run inference and save predictions with mixing metadata:
    python option_c_mixing_covariate_es_en_improved.py \
        --mode infer \
        --model gpt-4o-mini \
        --lang_pairs es-en \
        --max_samples 200

    # Analyze existing prediction CSVs, no new API calls:
    python option_c_mixing_covariate_es_en_improved.py \
        --mode analyze \
        --pred_dir result/option_c

    # Run both inference and analysis:
    python option_c_mixing_covariate_es_en_improved.py \
        --mode both \
        --model gpt-4o-mini \
        --lang_pairs es-en \
        --max_samples 200
"""

import os
import re
import math
import argparse
import asyncio
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    import warnings
    warnings.warn("statsmodels not installed; logistic regression will use sklearn.")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_DATASET = "CodeMixBench/CodeMixBench"

# User-facing language-pair names. These are mapped to the actual
# CodeMixBench Hugging Face config names below.
LANG_PAIRS = [
    "zh-en", "hi-en", "bn-en", "mr-en", "ne-en",
    "es-en", "fr-en", "ar-en", "ta-en", "nl-en", "de-en",
]

# CodeMixBench does not use config names like cm_mmlu_es_en.
# It uses names like mmlu_spaeng, mmlu_chieng, etc.
MMLU_CONFIG_MAP = {
    "zh-en": "mmlu_chieng",
    "hi-en": "mmlu_hineng",
    "bn-en": "mmlu_beneng",
    "mr-en": "mmlu_mareng",
    "ne-en": "mmlu_nepeng",
    "es-en": "mmlu_spaeng",
    "fr-en": "mmlu_freeng",
    "ar-en": "mmlu_araeng",
    "ta-en": "mmlu_tameng",
    "nl-en": "mmlu_duteng",
    "de-en": "mmlu_gereng",
}

CROSS_FAMILY_PAIRS = {"zh-en", "ar-en", "ta-en"}
SAME_BRANCH_PAIRS = {"de-en", "nl-en", "fr-en", "es-en"}
INDOARYAN_PAIRS = {"hi-en", "bn-en", "mr-en", "ne-en"}

FAMILY_LABEL = {
    "de-en": "Germanic",
    "nl-en": "Germanic",
    "fr-en": "Romance",
    "es-en": "Romance",
    "hi-en": "Indo-Aryan",
    "bn-en": "Indo-Aryan",
    "mr-en": "Indo-Aryan",
    "ne-en": "Indo-Aryan",
    "zh-en": "Sino-Tibetan",
    "ar-en": "Afro-Asiatic",
    "ta-en": "Dravidian",
}

FAMILY_COLOR = {
    "Germanic": "#2196F3",
    "Romance": "#4CAF50",
    "Indo-Aryan": "#FF9800",
    "Sino-Tibetan": "#E91E63",
    "Afro-Asiatic": "#9C27B0",
    "Dravidian": "#F44336",
}

EXCLUDED_LID_TAGS = {
    "other", "punct", "ne", "mixed", "ambiguous", "unk",
    "foreign", "lang-indep", "number", "num", "symbol",
}


# ---------------------------------------------------------------------------
# Spanish-English fallback language ID heuristic
# ---------------------------------------------------------------------------
SPANISH_WORDS = {
    # Articles / determiners
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas",

    # Prepositions / conjunctions
    "de", "del", "a", "al", "en", "por", "para", "con", "sin", "sobre",
    "entre", "hasta", "desde", "durante", "contra", "según",
    "y", "o", "pero", "aunque", "porque", "como", "cuando", "donde", "mientras",

    # Pronouns / clitics
    "yo", "tú", "tu", "usted", "él", "ella", "nosotros", "nosotras", "vosotros",
    "vosotras", "ellos", "ellas", "me", "te", "se", "nos", "os", "lo", "la", "le",
    "los", "las", "les", "mi", "mis", "tu", "tus", "su", "sus", "nuestro",
    "nuestra", "nuestros", "nuestras",

    # Auxiliaries / common verbs
    "es", "son", "soy", "eres", "somos", "fue", "fueron", "era", "eran", "ser",
    "estar", "está", "estan", "están", "estoy", "estaba", "estaban", "hay", "había",
    "habia", "tiene", "tienen", "tener", "tuvo", "hacer", "hace", "hacen", "hizo",
    "puede", "pueden", "poder", "debe", "deben", "deber", "dice", "dicen", "dijo",

    # Question / degree words
    "qué", "que", "cuál", "cual", "cuáles", "cuales", "quién", "quien", "quienes",
    "cuánto", "cuanto", "cuánta", "cuanta", "cuántos", "cuantos", "cuántas", "cuantas",
    "más", "mas", "menos", "muy", "también", "tambien", "solo", "sólo", "ya", "no", "sí", "si",

    # Common content words useful for MMLU-like questions
    "persona", "personas", "hombre", "mujer", "niño", "niña", "familia", "grupo",
    "sociedad", "comunidad", "gobierno", "estado", "país", "pais", "ciudad", "mundo",
    "historia", "cultura", "lengua", "idioma", "palabra", "texto", "pregunta", "respuesta",
    "opción", "opcion", "opciones", "ejemplo", "caso", "problema", "sistema", "proceso",
    "resultado", "causa", "efecto", "razón", "razon", "forma", "tipo", "parte", "número", "numero",
    "tiempo", "año", "ano", "día", "dia", "agua", "tierra", "aire", "energía", "energia",
    "fuerza", "masa", "velocidad", "temperatura", "ciencia", "biología", "biologia",
    "química", "quimica", "física", "fisica", "matemática", "matematica", "economía", "economia",
    "ley", "derecho", "mercado", "trabajo", "escuela", "universidad", "estudiante",
    "profesor", "profesora", "libro", "comida", "casa", "animal", "planta", "célula", "celula",
    "grande", "pequeño", "pequeno", "bueno", "buena", "malo", "mala", "nuevo", "nueva",
    "viejo", "vieja", "mayor", "menor", "primero", "segunda", "segundo", "principal",
}

ENGLISH_WORDS = {
    # Articles / determiners
    "the", "a", "an", "this", "that", "these", "those", "each", "every", "some", "any",

    # Prepositions / conjunctions
    "of", "to", "in", "on", "for", "with", "without", "by", "from", "at", "as", "into",
    "between", "through", "during", "before", "after", "and", "or", "but", "because",
    "although", "while", "where", "when",

    # Pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our", "its",

    # Auxiliaries / common verbs
    "is", "are", "am", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "may", "might", "must", "should", "would", "will",
    "there", "make", "makes", "made", "take", "takes", "took", "given", "choose",

    # Question / degree words
    "what", "which", "who", "whom", "whose", "why", "how", "more", "less", "very", "also",
    "only", "not", "yes", "no",

    # Common content words useful for MMLU-like questions
    "person", "people", "man", "woman", "child", "family", "group", "society", "community",
    "government", "state", "country", "city", "world", "history", "culture", "language",
    "word", "text", "question", "answer", "option", "example", "case", "problem", "system",
    "process", "result", "cause", "effect", "reason", "form", "type", "part", "number",
    "time", "year", "day", "water", "earth", "air", "energy", "force", "mass", "speed",
    "temperature", "science", "biology", "chemistry", "physics", "mathematics", "economics",
    "law", "market", "work", "school", "university", "student", "teacher", "book", "food",
    "house", "animal", "plant", "cell", "large", "small", "good", "bad", "new", "old",
    "greater", "lesser", "first", "second", "main",
}

SPANISH_SUFFIXES = (
    "ción", "ciones", "sión", "siones", "dad", "dades", "idades", "mente",
    "amiento", "amientos", "imiento", "imientos", "ado", "ada", "ados", "adas",
    "ido", "ida", "idos", "idas", "ando", "iendo", "ar", "er", "ir",
)

ENGLISH_SUFFIXES = (
    "tion", "tions", "sion", "sions", "ness", "ment", "ments", "ing",
    "ed", "ly", "ous", "ive", "al", "ity", "ities",
)


def clean_token_for_langid(token: str) -> str:
    """Normalize one token for fallback language identification."""
    token = str(token).lower().strip()
    token = re.sub(
        r"^[^\wáéíóúñü¿¡]+|[^\wáéíóúñü¿¡]+$",
        "",
        token,
        flags=re.IGNORECASE,
    )
    return token


def guess_latin_token_language(token: str, lang1: str, lang2: str) -> str:
    """
    Better fallback heuristic for same-script Latin pairs.

    This is strongest for es-en. It deliberately avoids the old behavior of
    labeling every ASCII token as English.
    """
    tok = clean_token_for_langid(token)

    if not tok:
        return "other"

    if re.fullmatch(r"[\d\W_]+", tok):
        return "other"

    # Spanish-specific characters are strong evidence for Spanish.
    if any(ch in tok for ch in "áéíóúñü¿¡"):
        return "es"

    # Main improvement: Spanish-English lexical and morphology heuristic.
    if {lang1, lang2} == {"es", "en"}:
        if tok in SPANISH_WORDS:
            return "es"
        if tok in ENGLISH_WORDS:
            return "en"

        if tok.endswith(SPANISH_SUFFIXES):
            return "es"
        if tok.endswith(ENGLISH_SUFFIXES):
            return "en"

        # Common Spanish conjugational endings.
        if re.search(r"(amos|emos|imos|aste|iste|aron|ieron|aba|aban|ía|ian|ían)$", tok):
            return "es"

        # Unknown Latin token. Do not assume English.
        return "other"

    # For fr-en, de-en, nl-en we leave unknown Latin tokens as other unless
    # you add dedicated lexicons/morphology for those languages.
    return "other"


# ---------------------------------------------------------------------------
# Mixing metric computation
# ---------------------------------------------------------------------------
def compute_m_index(lang_tags: list) -> float:
    """
    M-index = (1 - Σ pj²) / ((k-1) · Σ pj²)

    pj = proportion of language j, k = number of included languages.
    Excluded labels are removed before computing the score.
    """
    counts = Counter(
        str(t).lower()
        for t in lang_tags
        if t and str(t).lower() not in EXCLUDED_LID_TAGS
    )
    total = sum(counts.values())
    k = len(counts)

    if total == 0 or k < 2:
        return 0.0

    pj_sq_sum = sum((c / total) ** 2 for c in counts.values())
    if pj_sq_sum == 0:
        return 0.0

    return (1 - pj_sq_sum) / ((k - 1) * pj_sq_sum)


def compute_i_index(lang_tags: list) -> float:
    """
    I-index = number_of_switches / (n - 1)

    A switch is counted whenever adjacent included tokens differ in language.
    """
    filtered = [
        str(t).lower()
        for t in lang_tags
        if t and str(t).lower() not in EXCLUDED_LID_TAGS
    ]
    n = len(filtered)

    if n < 2:
        return 0.0

    switches = sum(1 for i in range(n - 1) if filtered[i] != filtered[i + 1])
    return switches / (n - 1)


def mixing_metrics_from_text(text: str, lang1: str, lang2: str) -> tuple:
    """
    Approximate M-index and I-index from raw text when per-token LID tags
    are unavailable.

    Priority:
      - Script-based detection for script-distinct language pairs.
      - Improved es-en Latin-script heuristic.
      - Unknown Latin tokens are labeled 'other', not automatically English.
    """
    LANG1_RANGES = {
        "zh": (0x4E00, 0x9FFF),      # CJK Unified Ideographs
        "ar": (0x0600, 0x06FF),      # Arabic
        "hi": (0x0900, 0x097F),      # Devanagari
        "mr": (0x0900, 0x097F),
        "ne": (0x0900, 0x097F),
        "bn": (0x0980, 0x09FF),      # Bengali
        "ta": (0x0B80, 0x0BFF),      # Tamil
        "ml": (0x0D00, 0x0D7F),      # Malayalam
    }

    lang1 = lang1.lower()
    lang2 = lang2.lower()
    l1_range = LANG1_RANGES.get(lang1)

    token_tags = []

    for word in str(text).split():
        if not word:
            continue

        chars = [ord(c) for c in word]

        # Script-distinct non-English language.
        if l1_range and any(l1_range[0] <= c <= l1_range[1] for c in chars):
            token_tags.append(lang1)
            continue

        # For script-distinct pairs, Latin letters are likely English.
        if l1_range and re.search(r"[A-Za-z]", word):
            token_tags.append("en")
            continue

        # Same-script Latin pairs, especially es-en.
        if not l1_range:
            guessed = guess_latin_token_language(word, lang1, lang2)
            token_tags.append(guessed)
            continue

        token_tags.append("other")

    return compute_m_index(token_tags), compute_i_index(token_tags)


def get_text_for_mixing(item: dict) -> str:
    """
    Choose the best available text field for fallback mixing estimation.
    """
    for key in ["question", "sentence", "text", "input", "prompt"]:
        if key in item and item[key]:
            return str(item[key])
    return ""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_cm_mmlu(lang_pair: str, split: str = "test"):
    from datasets import load_dataset as hf_load

    if lang_pair not in MMLU_CONFIG_MAP:
        raise ValueError(
            f"Unknown language pair: {lang_pair}. "
            f"Expected one of: {sorted(MMLU_CONFIG_MAP)}"
        )

    config = MMLU_CONFIG_MAP[lang_pair]

    try:
        ds = hf_load(HF_DATASET, config, split=split, trust_remote_code=True)
        log.info(f"Loaded {len(ds)} samples for {lang_pair} using config {config}.")
        return list(ds)
    except Exception as e:
        log.warning(f"HuggingFace load failed for {lang_pair} ({e}). Using stub.")
        return [
            {
                "question": f"Stub {i} ({lang_pair})?\n(A) A\n(B) B\n(C) C\n(D) D",
                "answer": "A",
                "tokens": [],
                "lid_tags": [],
            }
            for i in range(30)
        ]


def enrich_with_mixing_metrics(items: list, lang_pair: str) -> list:
    """
    For each item, compute M-index and I-index.

    Priority order:
      1. Use existing m_index/i_index columns if present.
      2. Use token-level language tags if available.
      3. Fall back to improved raw-text heuristic.
    """
    enriched = []
    l1, l2 = lang_pair.split("-", 1)

    for item in items:
        item = dict(item)

        if "m_index" in item and "i_index" in item:
            pass

        elif "lid_tags" in item and item["lid_tags"]:
            tags = item["lid_tags"]
            item["m_index"] = compute_m_index(tags)
            item["i_index"] = compute_i_index(tags)

        elif "language_tags" in item and item["language_tags"]:
            tags = item["language_tags"]
            item["m_index"] = compute_m_index(tags)
            item["i_index"] = compute_i_index(tags)

        elif "lang_tags" in item and item["lang_tags"]:
            tags = item["lang_tags"]
            item["m_index"] = compute_m_index(tags)
            item["i_index"] = compute_i_index(tags)

        else:
            text = get_text_for_mixing(item)
            m, i = mixing_metrics_from_text(text, l1, l2)
            item["m_index"] = m
            item["i_index"] = i

        enriched.append(item)

    return enriched


# ---------------------------------------------------------------------------
# OpenAI inference
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a system possessing knowledge in all subjects. "
    "You are skilled at selecting the correct answer based on multiple-choice questions. "
    "Do not include explanations in your answer."
)


def get_question_text(item: dict) -> str:
    """
    CodeMixBench MMLU-style configs often use 'sentence' rather than 'question'.
    This helper makes inference robust across both schemas.
    """
    for key in ["question", "sentence", "text", "input", "prompt"]:
        if key in item and item[key] is not None:
            return str(item[key])
    raise KeyError(f"No question-like text field found. Available keys: {list(item.keys())}")


def get_answer_label(item: dict) -> str:
    """
    Return the gold answer label as a clean string.
    """
    for key in ["answer", "label", "target", "gold"]:
        if key in item and item[key] is not None:
            return str(item[key]).strip().upper()
    return ""


def build_messages(examples: list, target: dict) -> list:
    msgs = []

    for ex in examples:
        msgs.append({"role": "user", "content": f"Question: {get_question_text(ex)}"})
        msgs.append({"role": "assistant", "content": f"Answer: {get_answer_label(ex)}"})

    msgs.append({"role": "user", "content": f"Question: {get_question_text(target)}"})
    return msgs


async def call_openai(
    client,
    messages: list,
    model: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_retries: int = 5,
) -> str:
    system = [{"role": "system", "content": SYSTEM_PROMPT}]

    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=system + messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=16,
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            wait = 2 ** attempt
            log.warning(
                f"OpenAI error (attempt {attempt + 1}/{max_retries}): {repr(e)}. "
                f"Retry in {wait}s."
            )
            await asyncio.sleep(wait)

    return ""


def parse_answer(raw: str) -> str:
    if not raw:
        return ""

    m = re.search(r"\b([ABCD])\b", raw.upper())
    return m.group(1) if m else raw.strip().upper()[:1]


async def run_inference(
    client,
    model: str,
    items: list,
    k: int,
    concurrency: int = 8,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> list:
    semaphore = asyncio.Semaphore(concurrency)

    async def process(item, idx):
        async with semaphore:
            pool = [x for j, x in enumerate(items) if j != idx]
            examples = pool[:k]
            messages = build_messages(examples, item)
            pred = await call_openai(
                client,
                messages,
                model,
                temperature=temperature,
                top_p=top_p,
            )
            label = get_answer_label(item)
            parsed = parse_answer(pred)

            return {
                **item,
                "index": idx,
                "prediction": pred,
                "parsed": parsed,
                "label": label,
                "correct": int(parsed == label),
                "k": k,
            }

    tasks = [process(item, i) for i, item in enumerate(items)]
    results = []

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), leave=False):
        results.append(await coro)

    results.sort(key=lambda x: x["index"])
    return results


# ---------------------------------------------------------------------------
# Binned accuracy analysis
# ---------------------------------------------------------------------------
def binned_accuracy(df: pd.DataFrame, metric_col: str, n_bins: int = 4) -> pd.DataFrame:
    """
    Bin samples by metric_col into n_bins quantile bins.
    Returns columns:
      bin, bin_label, bin_min, bin_max, accuracy, count, std_error
    """
    df = df.dropna(subset=[metric_col, "correct"]).copy()

    if df.empty:
        return pd.DataFrame()

    try:
        df["bin"] = pd.qcut(df[metric_col], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df[metric_col], bins=n_bins, labels=False, duplicates="drop")

    result = (
        df.groupby("bin", observed=True)
        .agg(
            accuracy=("correct", "mean"),
            count=("correct", "count"),
            bin_min=(metric_col, "min"),
            bin_max=(metric_col, "max"),
        )
        .reset_index()
    )

    result["accuracy"] *= 100
    result["std_error"] = result.apply(
        lambda r: 100
        * math.sqrt((r["accuracy"] / 100) * (1 - r["accuracy"] / 100) / max(r["count"], 1)),
        axis=1,
    )
    result["bin_label"] = result.apply(
        lambda r: f"[{r['bin_min']:.2f}, {r['bin_max']:.2f}]",
        axis=1,
    )

    return result


# ---------------------------------------------------------------------------
# Figure C1 & C2 — Binned accuracy vs. M-index / I-index
# ---------------------------------------------------------------------------
def plot_binned_accuracy(all_preds: pd.DataFrame, metric_col: str, metric_label: str, output_path: str):
    lang_pairs = sorted(all_preds["lang_pair"].unique())
    n = len(lang_pairs)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3), squeeze=False)
    axes_flat = axes.flatten()

    for idx, lp in enumerate(lang_pairs):
        ax = axes_flat[idx]
        sub = all_preds[all_preds["lang_pair"] == lp]
        binned = binned_accuracy(sub, metric_col)

        if binned.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            color = FAMILY_COLOR.get(FAMILY_LABEL.get(lp, ""), "steelblue")
            ax.bar(
                range(len(binned)),
                binned["accuracy"],
                yerr=binned["std_error"],
                color=color,
                alpha=0.8,
                capsize=4,
                error_kw={"elinewidth": 1},
            )
            ax.set_xticks(range(len(binned)))
            ax.set_xticklabels(binned["bin_label"], rotation=35, ha="right", fontsize=6)
            ax.set_ylim(0, 105)

            if len(binned) >= 2:
                rho, pval = spearmanr(range(len(binned)), binned["accuracy"])
                sign = "↓" if rho < 0 else "↑"
                sig = "*" if pval < 0.05 else ""
                ax.set_title(f"{lp}\n(ρ={rho:.2f}{sig} {sign})", fontsize=8, fontweight="bold")
            else:
                ax.set_title(lp, fontsize=8, fontweight="bold")

        if idx % ncols == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=8)
        ax.set_xlabel(metric_label, fontsize=7)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Accuracy vs. {metric_label} (CM-MMLU, binned by quartile)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info(f"Binned accuracy plot saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure C3 — Interaction: mixing intensity × language family
# ---------------------------------------------------------------------------
def plot_interaction(all_preds: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric_col, metric_label in [
        (axes[0], "m_index", "M-index"),
        (axes[1], "i_index", "I-index"),
    ]:
        groups = {
            "Cross-family (zh, ar, ta)": all_preds[all_preds["lang_pair"].isin(CROSS_FAMILY_PAIRS)],
            "Same-branch (de, nl, fr, es)": all_preds[all_preds["lang_pair"].isin(SAME_BRANCH_PAIRS)],
            "Indo-Aryan (hi, bn, mr, ne)": all_preds[all_preds["lang_pair"].isin(INDOARYAN_PAIRS)],
        }
        colors = ["#E91E63", "#2196F3", "#FF9800"]

        for (label, grp), color in zip(groups.items(), colors):
            binned = binned_accuracy(grp, metric_col, n_bins=4)
            if binned.empty:
                continue

            x = range(len(binned))
            ax.plot(x, binned["accuracy"], marker="o", linewidth=2, label=label, color=color)
            ax.fill_between(
                x,
                binned["accuracy"] - binned["std_error"],
                binned["accuracy"] + binned["std_error"],
                alpha=0.12,
                color=color,
            )
            ax.set_xticks(list(x))
            ax.set_xticklabels(binned["bin_label"], rotation=25, ha="right", fontsize=7)

        ax.set_xlabel(f"{metric_label} Quartile Bin", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_title(f"Accuracy vs. {metric_label}\nby Language Family Group", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(0, 105)
        ax.grid(linestyle=":", alpha=0.4)

    fig.suptitle("Interaction: Mixing Intensity × Language Family (CM-MMLU)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info(f"Interaction plot saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Scatter: M-index/I-index vs. per-instance correctness
# ---------------------------------------------------------------------------
def plot_mixing_vs_accuracy_scatter(df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric_col, metric_label in [
        (axes[0], "m_index", "M-index"),
        (axes[1], "i_index", "I-index"),
    ]:
        sub = df.dropna(subset=[metric_col, "correct"]).copy()

        for fam, fam_df in sub.groupby("family"):
            jitter = np.random.uniform(-0.02, 0.02, len(fam_df))
            ax.scatter(
                fam_df[metric_col],
                fam_df["correct"] + jitter,
                alpha=0.15,
                s=8,
                color=FAMILY_COLOR.get(fam, "gray"),
                label=fam,
            )

        binned = binned_accuracy(sub, metric_col, n_bins=10)
        if not binned.empty:
            bin_centers = (binned["bin_min"] + binned["bin_max"]) / 2
            ax.plot(bin_centers, binned["accuracy"] / 100, "k-", linewidth=2, label="Bin mean", zorder=5)

        if len(sub) >= 2 and sub[metric_col].nunique() > 1:
            rho, pval = spearmanr(sub[metric_col], sub["correct"])
            corr_text = f"ρ = {rho:.3f}, p = {pval:.4f}"
        else:
            corr_text = "ρ = NA, p = NA"

        ax.text(
            0.05,
            0.92,
            corr_text,
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(metric_label, fontsize=10)
        ax.set_ylabel("Correct (0/1 + jitter)", fontsize=10)
        ax.set_title(f"Per-instance {metric_label} vs. Correctness", fontsize=10, fontweight="bold")

        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=FAMILY_COLOR.get(f, "gray"),
                markersize=7,
                label=f,
            )
            for f in sub["family"].dropna().unique()
        ]
        handles.append(plt.Line2D([0], [0], color="k", linewidth=2, label="Bin mean"))
        ax.legend(handles=handles, fontsize=7, loc="lower right")
        ax.grid(linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info(f"Scatter plot saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------
def prepare_regression_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["m_index", "i_index", "correct", "lang_pair"]).copy()
    df["is_cross_family"] = df["lang_pair"].isin(CROSS_FAMILY_PAIRS).astype(int)
    df["is_indoaryan"] = df["lang_pair"].isin(INDOARYAN_PAIRS).astype(int)

    if df["m_index"].std() > 0:
        df["m_index_z"] = (df["m_index"] - df["m_index"].mean()) / df["m_index"].std()
    else:
        df["m_index_z"] = 0.0

    if df["i_index"].std() > 0:
        df["i_index_z"] = (df["i_index"] - df["i_index"].mean()) / df["i_index"].std()
    else:
        df["i_index_z"] = 0.0

    return df


def run_logistic_regression(df: pd.DataFrame):
    """
    Predict P(correct) from m_index, i_index, is_cross_family, is_indoaryan.
    """
    df = prepare_regression_df(df)

    features = ["m_index_z", "i_index_z", "is_cross_family", "is_indoaryan"]
    X = df[features]
    y = df["correct"]

    if len(df) < 10 or y.nunique() < 2:
        print("\n=== Logistic Regression skipped: not enough variation/data ===")
        return None

    if HAS_STATSMODELS:
        X_const = sm.add_constant(X)
        model = sm.Logit(y, X_const).fit(disp=0)
        print("\n=== Logistic Regression (statsmodels) ===")
        print(model.summary2().tables[1][["Coef.", "Std.Err.", "z", "P>|z|"]])
        return model

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_scaled, y)
    coef_df = pd.DataFrame({"feature": features, "coefficient": lr.coef_[0]}).sort_values(
        "coefficient", key=abs, ascending=False
    )
    print("\n=== Logistic Regression (sklearn — no p-values) ===")
    print(coef_df.to_string(index=False))
    return lr


def run_interaction_logistic_regression(df: pd.DataFrame):
    """
    Stronger test: do M-index/I-index matter more for cross-family pairs?
    """
    if not HAS_STATSMODELS:
        print("\nInteraction logistic regression requires statsmodels.")
        return None

    df = prepare_regression_df(df)

    if len(df) < 10 or df["correct"].nunique() < 2:
        print("\n=== Interaction Logistic Regression skipped: not enough variation/data ===")
        return None

    formula = (
        "correct ~ m_index_z * is_cross_family "
        "+ i_index_z * is_cross_family "
        "+ is_indoaryan"
    )

    model = sm.Logit.from_formula(formula, data=df).fit(disp=0)
    print("\n=== Interaction Logistic Regression ===")
    print(model.summary2().tables[1][["Coef.", "Std.Err.", "z", "P>|z|"]])
    return model


# ---------------------------------------------------------------------------
# Load existing prediction CSVs
# ---------------------------------------------------------------------------
def load_pred_csvs(pred_dir: str) -> pd.DataFrame:
    """
    Load all *_pred*.csv files from pred_dir that contain m_index and i_index.
    """
    dfs = []

    for fpath in Path(pred_dir).glob("**/*_pred*.csv"):
        try:
            df = pd.read_csv(fpath)

            if "lang_pair" not in df.columns:
                # Works for filenames like gpt-4o-mini_es_en_k1_pred_mixing.csv
                name = fpath.name.replace("_", "-")
                m = re.search(r"([a-z]{2,3}-en)", name)
                df["lang_pair"] = m.group(1) if m else "unknown"

            if "model" not in df.columns:
                df["model"] = fpath.stem.split("_")[0]

            if "m_index" in df.columns and "i_index" in df.columns:
                dfs.append(df)
                log.info(f"Loaded {len(df)} rows from {fpath.name}")
            else:
                log.debug(f"Skipping {fpath.name}; no mixing columns")

        except Exception as e:
            log.warning(f"Could not read {fpath}: {e}")

    if not dfs:
        log.warning("No prediction CSVs with mixing metadata found.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined["family"] = combined["lang_pair"].map(FAMILY_LABEL).fillna("Unknown")
    return combined


# ---------------------------------------------------------------------------
# Analysis mode
# ---------------------------------------------------------------------------
def run_analysis_mode(args):
    all_preds = load_pred_csvs(args.pred_dir)

    if all_preds.empty:
        log.error("No prediction data found. Run with --mode infer first.")
        return

    log.info(f"Total predictions loaded: {len(all_preds)}")
    log.info(f"Language pairs: {sorted(all_preds['lang_pair'].unique())}")
    log.info(f"M-index range: [{all_preds['m_index'].min():.3f}, {all_preds['m_index'].max():.3f}]")
    log.info(f"I-index range: [{all_preds['i_index'].min():.3f}, {all_preds['i_index'].max():.3f}]")

    summary = (
        all_preds.groupby("lang_pair")
        .agg(
            accuracy=("correct", lambda x: round(x.mean() * 100, 2)),
            mean_m_index=("m_index", "mean"),
            mean_i_index=("i_index", "mean"),
            count=("correct", "count"),
        )
        .reset_index()
    )

    print("\n=== Per-language-pair summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(args.pred_dir, "summary_by_langpair.csv"), index=False)

    print("\n=== Spearman: M-index / I-index vs. Correct ===")
    corr_rows = []

    for lp in sorted(all_preds["lang_pair"].unique()):
        sub = all_preds[all_preds["lang_pair"] == lp].dropna(subset=["m_index", "i_index", "correct"])

        if len(sub) < 10 or sub["correct"].nunique() < 2:
            continue

        rho_m, p_m = spearmanr(sub["m_index"], sub["correct"])
        rho_i, p_i = spearmanr(sub["i_index"], sub["correct"])

        corr_rows.append(
            {
                "lang_pair": lp,
                "rho_m_index": round(rho_m, 3),
                "p_m": round(p_m, 4),
                "rho_i_index": round(rho_i, 3),
                "p_i": round(p_i, 4),
            }
        )

    corr_df = pd.DataFrame(corr_rows)
    print(corr_df.to_string(index=False) if not corr_df.empty else "No valid correlations.")
    corr_df.to_csv(os.path.join(args.pred_dir, "spearman_mixing_vs_correct.csv"), index=False)

    plot_binned_accuracy(
        all_preds,
        metric_col="m_index",
        metric_label="M-index",
        output_path=os.path.join(args.pred_dir, "fig_c1_binned_m_index.png"),
    )
    plot_binned_accuracy(
        all_preds,
        metric_col="i_index",
        metric_label="I-index",
        output_path=os.path.join(args.pred_dir, "fig_c2_binned_i_index.png"),
    )
    plot_interaction(
        all_preds,
        output_path=os.path.join(args.pred_dir, "fig_c3_interaction.png"),
    )
    plot_mixing_vs_accuracy_scatter(
        all_preds,
        output_path=os.path.join(args.pred_dir, "fig_c4_scatter_mixing_vs_correct.png"),
    )

    run_logistic_regression(all_preds)
    run_interaction_logistic_regression(all_preds)


# ---------------------------------------------------------------------------
# Inference mode
# ---------------------------------------------------------------------------
async def run_inference_mode(args):
    if not HAS_OPENAI:
        raise ImportError("openai package not installed. Run: pip install openai")

    client = AsyncOpenAI(api_key=args.api_key)
    os.makedirs(args.pred_dir, exist_ok=True)

    for lang in args.lang_pairs:
        tag = f"{args.model.replace('/', '_')}_{lang.replace('-', '_')}_k{args.k_shot}"
        out_path = os.path.join(args.pred_dir, f"{tag}_pred_mixing.csv")

        if os.path.exists(out_path) and not args.force_rerun:
            log.info(f"[RESUME] {out_path} already exists, skipping.")
            continue

        log.info(f"\nRunning inference: {lang} | k={args.k_shot}")

        raw = load_cm_mmlu(lang)[: args.max_samples]
        items = enrich_with_mixing_metrics(raw, lang)

        results = await run_inference(
            client,
            args.model,
            items,
            k=args.k_shot,
            concurrency=args.concurrency,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        rows = []
        for r in results:
            rows.append(
                {
                    "index": r["index"],
                    "lang_pair": lang,
                    "model": args.model,
                    "k": r["k"],
                    "prediction": r.get("prediction", ""),
                    "parsed": r.get("parsed", ""),
                    "label": r.get("label", ""),
                    "correct": r.get("correct", 0),
                    "m_index": r.get("m_index", float("nan")),
                    "i_index": r.get("i_index", float("nan")),
                    "family": FAMILY_LABEL.get(lang, "Unknown"),
                    "question": get_question_text(r)[:500],
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        acc = df["correct"].mean() * 100 if len(df) else 0.0
        log.info(f"Saved {len(df)} rows to {out_path} | accuracy={acc:.2f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Option C: Mixing intensity as covariate")

    p.add_argument(
        "--mode",
        choices=["infer", "analyze", "both"],
        default="both",
        help="infer = run OpenAI and save CSVs; analyze = load CSVs and plot",
    )
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument(
        "--lang_pairs",
        nargs="+",
        default=LANG_PAIRS,
        help="Language pairs to evaluate, e.g. --lang_pairs es-en zh-en",
    )
    p.add_argument("--k_shot", type=int, default=1, help="Number of few-shot examples")
    p.add_argument("--max_samples", type=int, default=200, help="Max samples per language pair")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--pred_dir", default="result/option_c", help="Directory for prediction CSVs and figures")
    p.add_argument("--force_rerun", action="store_true")

    return p.parse_args()


async def main():
    args = parse_args()
    os.makedirs(args.pred_dir, exist_ok=True)

    if args.mode in ("infer", "both"):
        if not args.api_key:
            raise ValueError("Provide --api_key or set OPENAI_API_KEY")
        await run_inference_mode(args)

    if args.mode in ("analyze", "both"):
        run_analysis_mode(args)

    log.info(f"\nDone. Outputs in: {args.pred_dir}")


if __name__ == "__main__":
    asyncio.run(main())
