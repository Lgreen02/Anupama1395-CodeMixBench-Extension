import os
import argparse
import asyncio

from utils import load_dataset_from_hf, loadDataset, generate_Token_Label_Prompts, generate_Sentence_Label_Prompts, generate_MMLU_Prompts, generate_GSM8K_Prompts, generate_TruthfulQA_Prompts
from utils import loadAnswerFile, compute_token_label_Metric, compute_sentence_label_Metric, compute_BLEU
from utils import asyncAskGPT, asyncAskReplicate, analyse_Token_Label_Result, analyse_Sentence_Label_Result
import pandas as pd


def main() -> None:

    
    dataset = args.dataset
    model = args.model
    count = args.count
    shot = args.shot
    temperature = args.temperature
    top_p = args.top_p
    if api:
        api = args.api
    else:
        url = os.environ.get("OPENAI_API_KEY")

    if url:
        url = args.url
    else:
        url = os.environ.get("OPENAI_API_BASE")
    max_tokens = args.max_tokens

    #----------- parse task -------------
    if '_' in dataset:
        task, substr = dataset.split('_',1)


    #----------- create dir of the experiment -------------
    if args.expid:
        expid = args.expid
    else:
        expid = dataset + '_all'
    
    dir = f"./result/{dataset}"
    if not os.path.exists(dir):
        os.mkdir(dir)

    print(f"--dataset: {dataset}")
    print(f"--model: {model}")
    print(f"--experiment ID: {expid}")
    print(f"--experiment dir: {dir}")
    
    model_in_path = model.replace('.','_').replace('/','_')
    answerFilePath = f"{dir}/{model_in_path}_{expid}_answer_raw.csv"    # save complete answer from model to this file
    predFilePath = f"{dir}/{model_in_path}_{expid}_pred.csv"        # save expid's predicted value to this file
    matric_path = f"{dir}/{model_in_path}_{expid}_matric.txt"       # save expid's matric value to this file
    rawPath = answerFilePath
    if args.analyse:
        answerFilePath = f"{dir}/{model_in_path}_{expid}_answer_{args.analyse}.csv"


    #------------- load dataset from huggingface-------------
    df = load_dataset_from_hf(
        dataset = dataset,
        path = f"{task}/{dataset}.csv"
        )
    if task in 'lid, pos, ner':
        true_df = df[['index', 'tokens', 'answer']]        #改命名
    elif task in 'sa, mt, mmlu, gsm8k, truthfulqa':
        true_df = df[['index', 'sentence','answer']]


    #------------- Generate prompts -----------------
    if task in 'lid, pos, ner':
        promptsList = generate_Token_Label_Prompts(
            dataset= dataset,
            model = model,
            df = df
            )
    elif task in 'sa, mt':
        promptsList = generate_Sentence_Label_Prompts(
            dataset= dataset,
            model = model,
            df = df
            )
    elif task in 'mmlu':
        promptsList = generate_MMLU_Prompts(
            dataset= dataset,
            model = model,
            df = df,
            shot = shot
            )
    elif task in 'gsm8k':
        promptsList = generate_GSM8K_Prompts(
            dataset= dataset,
            model = model,
            df = df,
            shot = shot
            )
    elif task in 'truthfulqa':
        promptsList = generate_TruthfulQA_Prompts(
            dataset= dataset,
            model = model,
            df = df,
            shot = shot
            )
    else:
        print(f"no such task: {task}")
        exit()
    
    # Test some data in the dataset, default is to test the full dataset.
    if args.indices:
        indices = eval(args.indices)
    elif args.j:
        i = 0
        if args.i:
            i = int(args.i)
        j = int(args.j)
        indices = list(range(i, j))
    else:
        i = 0
        j = len(promptsList)
        indices = list(range(i, j))

    messageList = promptsList

    #------------- chat with Replicate API -------------
    #Because some models are only available through Replicate, special handling is required for the interfaces provided by Replicate.
    if model == "meta/meta-llama-3-8b" or model == "meta/meta-llama-3-70b" or model.startswith('meta/llama-2-'):
        if os.path.exists(answerFilePath):
            results = loadAnswerFile(answerFilePath)
        else:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())     #solve "RuntimeError: Event loop is closed" on Windows
            results = asyncio.run(
                asyncAskReplicate(
                    answerFilePath = answerFilePath,
                    messageList = messageList,
                    indices = indices,
                    model = model,
                    temperature=temperature,
                    top_p=top_p,
                    api_key=api,
                    max_tokens=max_tokens
                    )
                )
    else:
        #------------- chat with OpenAI API -------------
        # check the answerFilePath to see if already asked GPT and got the raw answer file
        if os.path.exists(answerFilePath):
            results = loadAnswerFile(answerFilePath)

        else:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())     #solve "RuntimeError: Event loop is closed" on Windows
            results = asyncio.run(
                asyncAskGPT(
                    answerFilePath = answerFilePath,
                    messageList = messageList,
                    indices = indices,
                    model = model,
                    temperature=temperature,
                    top_p=top_p,
                    api_key=api,
                    base_url=url,
                    max_tokens=max_tokens
                    )
                )

    #------------- analyse the result -------------
    results = loadAnswerFile(answerFilePath)
    if os.path.exists(predFilePath):
        pred_df = pd.read_csv(predFilePath)
        print(f"Get {len(pred_df)} pieces of answers from {predFilePath}.")

    else:
        if task in 'lid, pos, ner':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            pred_df = asyncio.run(
                analyse_Token_Label_Result(
                    results = results,
                    answerFilePath = answerFilePath,
                    predFilePath = predFilePath,
                    messageList = messageList,
                    model = model,
                    temperature=temperature,
                    top_p=top_p,
                    true_df=true_df,
                    again = count,
                    rawPath = rawPath,
                    api_key=api,
                    base_url=url
                    )
                )
        elif task in 'sa, mt, mmlu, gsm8k, truthfulqa':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            pred_df = asyncio.run(
                analyse_Sentence_Label_Result(
                    results = results,
                    answerFilePath = answerFilePath,
                    predFilePath = predFilePath,
                    messageList = messageList,
                    task = task,
                    model = model,
                    temperature=temperature,
                    top_p=top_p,
                    true_df=true_df,
                    again = count,
                    api_key=api,
                    base_url=url
                    )
                )

    #------------- load metric -------------
    if not os.path.exists(predFilePath):
        print(f"No clear predFilePath")
        exit()
    if task in 'lid, pos, ner':
        pred_df = compute_token_label_Metric(pred_df, matric_path)
        pred_df.to_csv(predFilePath.replace(".csv", "_metric.csv"), index=False, encoding='utf-8')

    elif task in 'sa, mmlu, gsm8k, truthfulqa':
        pred_df = compute_sentence_label_Metric(pred_df, matric_path)
    elif task in 'mt':
        pred_df = compute_BLEU(pred_df, expid, matric_path)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True, type=str, help="dataset name. (e.g., lid_gereng, lid_spaeng, ner_hineng)")
    parser.add_argument("--expid", type=str, help="The ID of this experiment, the results file will be named after this ID.")
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str, help="The model you chat with. Default model is gpt-3.5-turbo.")
    parser.add_argument("--indices", type=str, help="input a index list [1,2,3]")
    parser.add_argument("--i", type=int, help="number i of indices [i, j]")
    parser.add_argument("--j", type=int, help="number j of indices [i, j]")
    parser.add_argument("--count", default="3", type=int, help="ask gpt count times if any anwser has error")
    parser.add_argument("--analyse", type=str, help="analyse file")
    parser.add_argument("--shot", default=1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--top_p", default=0, type=float)
    parser.add_argument("--api", default=None, type=str)
    parser.add_argument("--url", default=None, type=str)
    parser.add_argument("--max_tokens", default=1024, type=int)


    args = parser.parse_args()

    
    main()


