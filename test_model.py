import os
import argparse
import asyncio
import ast

import pandas as pd

from utils import (
    load_dataset_from_hf,
    loadDataset,
    generate_Token_Label_Prompts,
    generate_Sentence_Label_Prompts,
    generate_MMLU_Prompts,
    generate_GSM8K_Prompts,
    generate_TruthfulQA_Prompts,
)

from utils import (
    loadAnswerFile,
    compute_token_label_Metric,
    compute_sentence_label_Metric,
    compute_BLEU,
)

from utils import (
    asyncAskGPT,
    asyncAskReplicate,
    analyse_Token_Label_Result,
    analyse_Sentence_Label_Result,
)


def run_async(coro):
    """
    Runs async code safely on Linux, macOS, and Windows.
    The original code used WindowsSelectorEventLoopPolicy unconditionally,
    which can fail on Linux.
    """
    if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    return asyncio.run(coro)


def main() -> None:
    dataset = args.dataset
    model = args.model
    count = args.count
    shot = args.shot
    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.max_tokens

    # Correct API and URL handling
    api = args.api or os.environ.get("OPENAI_API_KEY")
    url = args.url or os.environ.get("OPENAI_API_BASE")

    # If no custom API base URL is used, keep it as None
    if url == "":
        url = None

    # Parse task name from dataset name
    # Example: lid_guaspa -> task = lid
    if "_" not in dataset:
        print(f"Invalid dataset name: {dataset}")
        print("Dataset name should look like lid_guaspa, lid_gereng, ner_hineng, etc.")
        exit()

    task, substr = dataset.split("_", 1)

    # Create result directory
    expid = args.expid if args.expid else dataset + "_all"

    result_root = "./result"
    os.makedirs(result_root, exist_ok=True)

    experiment_dir = f"{result_root}/{dataset}"
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"--dataset: {dataset}")
    print(f"--task: {task}")
    print(f"--model: {model}")
    print(f"--experiment ID: {expid}")
    print(f"--experiment dir: {experiment_dir}")

    model_in_path = model.replace(".", "_").replace("/", "_")

    answerFilePath = f"{experiment_dir}/{model_in_path}_{expid}_answer_raw.csv"
    predFilePath = f"{experiment_dir}/{model_in_path}_{expid}_pred.csv"
    matric_path = f"{experiment_dir}/{model_in_path}_{expid}_matric.txt"
    rawPath = answerFilePath

    if args.analyse:
        answerFilePath = f"{experiment_dir}/{model_in_path}_{expid}_answer_{args.analyse}.csv"

    # Load dataset from Hugging Face
    df = load_dataset_from_hf(
        dataset=dataset,
        path=f"{task}/{dataset}.csv"
    )

    # Select ground-truth columns
    if task in ("lid", "pos", "ner"):
        true_df = df[["index", "tokens", "answer"]]
    elif task in ("sa", "mt", "mmlu", "gsm8k", "truthfulqa"):
        true_df = df[["index", "sentence", "answer"]]
    else:
        print(f"No such task: {task}")
        exit()

    # Generate prompts
    if task in ("lid", "pos", "ner"):
        promptsList = generate_Token_Label_Prompts(
            dataset=dataset,
            model=model,
            df=df
        )

    elif task in ("sa", "mt"):
        promptsList = generate_Sentence_Label_Prompts(
            dataset=dataset,
            model=model,
            df=df
        )

    elif task == "mmlu":
        promptsList = generate_MMLU_Prompts(
            dataset=dataset,
            model=model,
            df=df,
            shot=shot
        )

    elif task == "gsm8k":
        promptsList = generate_GSM8K_Prompts(
            dataset=dataset,
            model=model,
            df=df,
            shot=shot
        )

    elif task == "truthfulqa":
        promptsList = generate_TruthfulQA_Prompts(
            dataset=dataset,
            model=model,
            df=df,
            shot=shot
        )

    else:
        print(f"No such task: {task}")
        exit()

    # Choose which dataset rows to test
    # Full dataset by default. Use --j 2 for a tiny test.
    if args.indices:
        indices = ast.literal_eval(args.indices)
    elif args.j is not None:
        i = int(args.i) if args.i is not None else 0
        j = int(args.j)
        indices = list(range(i, j))
    else:
        i = 0
        j = len(promptsList)
        indices = list(range(i, j))

    messageList = promptsList

    # Ask model
    is_replicate_model = (
        model == "meta/meta-llama-3-8b"
        or model == "meta/meta-llama-3-70b"
        or model.startswith("meta/llama-2-")
    )

    if is_replicate_model:
        # Replicate models
        if os.path.exists(answerFilePath):
            results = loadAnswerFile(answerFilePath)
        else:
            results = run_async(
                asyncAskReplicate(
                    answerFilePath=answerFilePath,
                    messageList=messageList,
                    indices=indices,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    api_key=api,
                    max_tokens=max_tokens
                )
            )

    else:
        # OpenAI-compatible models
        if os.path.exists(answerFilePath):
            results = loadAnswerFile(answerFilePath)
        else:
            if not api:
                raise ValueError(
                    "No API key found. Pass it using --api YOUR_KEY "
                    "or set OPENAI_API_KEY in your environment."
                )

            results = run_async(
                asyncAskGPT(
                    answerFilePath=answerFilePath,
                    messageList=messageList,
                    indices=indices,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    api_key=api,
                    base_url=url,
                    max_tokens=max_tokens
                )
            )

    # Analyse result
    results = loadAnswerFile(answerFilePath)

    if os.path.exists(predFilePath):
        pred_df = pd.read_csv(predFilePath)
        print(f"Get {len(pred_df)} pieces of answers from {predFilePath}.")

    else:
        if task in ("lid", "pos", "ner"):
            pred_df = run_async(
                analyse_Token_Label_Result(
                    results=results,
                    answerFilePath=answerFilePath,
                    predFilePath=predFilePath,
                    messageList=messageList,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    true_df=true_df,
                    again=count,
                    rawPath=rawPath,
                    api_key=api,
                    base_url=url
                )
            )

        elif task in ("sa", "mt", "mmlu", "gsm8k", "truthfulqa"):
            pred_df = run_async(
                analyse_Sentence_Label_Result(
                    results=results,
                    answerFilePath=answerFilePath,
                    predFilePath=predFilePath,
                    messageList=messageList,
                    task=task,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    true_df=true_df,
                    again=count,
                    api_key=api,
                    base_url=url
                )
            )

        else:
            print(f"No such task during analysis: {task}")
            exit()

    # Compute metric
    if not os.path.exists(predFilePath):
        print("No clear predFilePath")
        exit()

    if task in ("lid", "pos", "ner"):
        pred_df = compute_token_label_Metric(pred_df, matric_path)
        pred_df.to_csv(
            predFilePath.replace(".csv", "_metric.csv"),
            index=False,
            encoding="utf-8"
        )

    elif task in ("sa", "mmlu", "gsm8k", "truthfulqa"):
        compute_sentence_label_Metric(pred_df, matric_path)

    elif task == "mt":
        compute_BLEU(pred_df, expid, matric_path)

    print("Done.")
    print(f"Raw answers saved to: {answerFilePath}")
    print(f"Predictions saved to: {predFilePath}")
    print(f"Metrics saved to: {matric_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="Dataset name, e.g., lid_gereng, lid_spaeng, ner_hineng"
    )

    parser.add_argument(
        "--expid",
        type=str,
        help="Experiment ID. Result files will be named after this ID."
    )

    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        type=str,
        help="Model name. Default is gpt-3.5-turbo."
    )

    parser.add_argument(
        "--indices",
        type=str,
        help="Input an index list, e.g., '[1,2,3]'"
    )

    parser.add_argument(
        "--i",
        type=int,
        help="Start index for range [i, j]"
    )

    parser.add_argument(
        "--j",
        type=int,
        help="End index for range [i, j]"
    )

    parser.add_argument(
        "--count",
        default=3,
        type=int,
        help="Ask model again this many times if any answer has error"
    )

    parser.add_argument(
        "--analyse",
        type=str,
        help="Analyse file suffix"
    )

    parser.add_argument(
        "--shot",
        default=1,
        type=int
    )

    parser.add_argument(
        "--temperature",
        default=0,
        type=float
    )

    parser.add_argument(
        "--top_p",
        default=0,
        type=float
    )

    parser.add_argument(
        "--api",
        default=None,
        type=str,
        help="OpenAI API key or Replicate API token"
    )

    parser.add_argument(
        "--url",
        default=None,
        type=str,
        help="Optional custom OpenAI-compatible API base URL"
    )

    parser.add_argument(
        "--max_tokens",
        default=1024,
        type=int
    )

    args = parser.parse_args()

    main()