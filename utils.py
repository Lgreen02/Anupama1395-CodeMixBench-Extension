import os
import openai
from openai import OpenAI

import pandas as pd
from pandas import DataFrame as df
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sacrebleu.metrics import BLEU
from datasets import load_dataset

import time
import json
import re
import logging
from logging import Logger
import copy
import ast

from asyncio import run
from oaib import Batch, BatchReplicate


def getMyLogger(loggerName:str, filePath:str="./log.log", loggingLevel:int=logging.DEBUG) -> Logger:
    # create logger
    logger = logging.getLogger(loggerName)
    logger.setLevel(loggingLevel)

    if not logger.hasHandlers():
        # create file&stream handler and set level to DEBUG(10)
        fh = logging.FileHandler(filePath, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setLevel(loggingLevel)

        # create formatter, and add formatter to fh,sh
        formatter = logging.Formatter(
            datefmt="%Y-%m-%d %H:%M:%S",
            fmt='%(asctime)s %(name)s [%(filename)s <%(funcName)s> line:%(lineno)d] %(levelname)s: %(message)s'
            )
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)

        # add fh,sh to logger
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.info(f"MyLogger {loggerName} initialized at level {loggingLevel}")
    else:
        logger.info(f"Use existed MyLogger {loggerName} at level {loggingLevel}")

    return logger
logger = getMyLogger('fileLogger', loggingLevel=logging.DEBUG)


def generate_Token_Label_Prompts(dataset:str, model:str, df:df=None) -> list:
    #logger = getMyLogger('fileLogger','log.log')

    # get template from  'prompt.json'
    template = getTemplate(dataset)
    #rawContent = template[1]['content']
    
    promptsList = []
    for index, row in df.iterrows():
        #jsonFormat = list2json(row['tokens'])
        if model == 'gpt-3.5-turbo-instruct':
            prompt = f"{template[0]['content']} Tokenized sentence: {row['tokens']}; Your final answer:"
        else:
            template[1]['content'] = f"Tokenized sentence: {row['tokens']}; Your final answer:"
            prompt = copy.deepcopy(template)
        promptsList.append(prompt)
        
        #print(prompt)
        
    logger.info(f"Genarate {len(promptsList)} pieces of prompts")
    return promptsList

def generate_Sentence_Label_Prompts(dataset:str, model:str, df:df=None) -> list:
    #logger = getMyLogger('fileLogger','log.log')

    # get template from  'prompt.json'
    template = getTemplate(dataset)
    #rawContent = template[1]['content']
    
    promptsList = []
    for index, row in df.iterrows():
        #jsonFormat = list2json(row['tokens'])
        if model == 'gpt-3.5-turbo-instruct':
            prompt = f"{template[0]['content']}\nSentence: {row['sentence']}; Your final answer:"
        else:
            template[1]['content'] = f"Sentence: {row['sentence']}; Your final answer:"
            prompt = copy.deepcopy(template)
        promptsList.append(prompt)
        
        #print(prompt)
        
    logger.info(f"Genarate {len(promptsList)} pieces of prompts")
    return promptsList

def generate_MMLU_Prompts(dataset:str, model:str, df:df=None, shot:int=1) -> list:
    #logger = getMyLogger('fileLogger','log.log')

    # get template from  'prompt.json'
    template = getTemplate(dataset)
    #rawContent = template[1]['content']
    
    promptsList = []
    for index, row in df.iterrows():
        #jsonFormat = list2json(row['tokens'])

        if model == 'gpt-3.5-turbo-instruct' or model == "meta/meta-llama-3-8b" or model == "meta/meta-llama-3-70b" or model.startswith('meta/llama-2-'):
            shot_number = shot
            examples_str = ""
            while shot_number > 0:
                random_row = df.sample(n=1)
                while len(random_row['sentence'].values[0]) > 600:
                    random_row = df.sample(n=1)
                if row['index'] == random_row['index'].values[0]:
                    continue
                if random_row['sentence'].values[0] in examples_str:
                    continue
                example_str = f"{random_row['sentence'].values[0]}\nAnswer: {random_row['answer'].values[0]}\n"
                examples_str += example_str
                shot_number-=1

            prompt = f"{template[0]['content']}\n{examples_str}\n{row['sentence']}\nAnswer:"
            
        else:
            shot_number = 0
            prompt = [{
                "role": "system", "content": f"{template[0]['content']}"
            }]
            while shot_number < shot:
                
                random_row = df.sample(n=1)
                while len(random_row['sentence'].values[0]) > 600:
                    random_row = df.sample(n=1)
                if row['index'] == random_row['index'].values[0]:
                    continue
                """ if random_row['sentence'].values[0] in examples_str:
                    continue """
                prompt.append({
                        "role": "user",
                        "content": f"{random_row['sentence'].values[0]}\n\nAnswer:"
                    })
                prompt.append({
                        "role": "assistant",
                        "content": f"{random_row['answer'].values[0]}"
                    })
                shot_number += 1

            prompt.append({
                        "role": "user",
                        "content": f"{row['sentence']}\n\nAnswer:"
                    })
            #template[1]['content'] = f"{examples_str}\n{row['sentence']}\nAnswer:"
            
        #prompt = copy.deepcopy(template)
        promptsList.append(prompt)
        
        #print(prompt)
        
    logger.info(f"Genarate {len(promptsList)} pieces of prompts")
    return promptsList

def generate_GSM8K_Prompts(dataset:str, model:str, df:df=None, shot:int=1) -> list:
    #logger = getMyLogger('fileLogger','log.log')

    # get template from  'prompt.json'
    template = getTemplate(dataset)
    #rawContent = template[1]['content']

    
    promptsList = []
    for index, row in df.iterrows():
        #jsonFormat = list2json(row['tokens'])

        if model == 'gpt-3.5-turbo-instruct' or model == "meta/meta-llama-3-8b" or model == "meta/meta-llama-3-70b" or model.startswith('meta/llama-2-'):
            shot_number = shot
            examples_str = ""
            while shot_number > 0:
                random_row = df.sample(n=1)
                if row['index'] == random_row['index'].values[0]:
                    continue
                example_str = f"Problem: {random_row['sentence'].values[0]}\nSolution: {random_row['cot'].values[0]}\nFinal Answer: {random_row['answer'].values[0]} [stop]\n\n"
                examples_str += example_str
                shot_number-=1
            if shot == 0:
                prompt = f"{template[0]['content']}\nOutput the solution and final answer for the next problem. The solution should include the entire process of calculating the final answer. The final answer to the problem is just one definite numerical value. Don't output the problem. Output in this format:\nSolution:\nFinal answer: (one definite numerical value) [stop]\n\nProblem: {row['sentence']}"
            else:
                prompt = f"{template[0]['content']}\n\nComplete the solution and final answer for the next problem. The solution should include the entire process of calculating the final answer. The final answer to the problem is just one definite numerical value. Don't output the problem. \n\n{examples_str}\n\nProblem: {row['sentence']}"
                #prompt = f"{template[0]['content']}\nExamples:\n\n{examples_str}\nRead and follow the format of the examples above. Output the solution and final answer for the next problem. The solution should include the entire process of calculating the final answer. The final answer to the problem is just one definite numerical value. Don't output the problem. Output the solution and the final answer in this format:\nSolution:\nFinal answer: (one definite numerical value) [stop]\n\nProblem: {row['sentence']}"
        else:
            """ if shot == 0:
                template[1]['content'] = f"Output the solution and final answer for the next problem. The solution should include the entire process of calculating the final answer. The final answer to the problem is just one definite numerical value. Don't output the problem. Output in this format:\nSolution:\nFinal answer: (one definite numerical value)\n\nProblem: {row['sentence']}"
            else: """
            shot_number = 0
            prompt = [{
                "role": "system", "content": f"{template[0]['content']}\nOutput the solution and final answer for the next problem. The solution should include the entire process of calculating the final answer. The final answer to the problem is just one definite numerical value. Don't output the problem. Output in this format:\nSolution:\nFinal answer: (one definite numerical value)"
            }]
            while shot_number < shot:
                
                random_row = df.sample(n=1)
                while len(random_row['sentence'].values[0]) > 600:
                    random_row = df.sample(n=1)
                if row['index'] == random_row['index'].values[0]:
                    continue
                """ if random_row['sentence'].values[0] in examples_str:
                    continue """
                prompt.append({
                        "role": "user",
                        "content": f"Problem:\n{random_row['sentence'].values[0]}"
                    })
                prompt.append({
                        "role": "assistant",
                        "content": f"Solution:\n{random_row['cot'].values[0]}\n\n\nFinal Answer: {random_row['answer'].values[0]}"
                    })
                shot_number += 1
            #template[1]['content'] = f"Examples:\n\n{examples_str}\nRead and follow the format of the examples above. Output the solution and final answer for the next problem. The solution should include the entire process of calculating the final answer. The final answer to the problem is just one definite numerical value. Don't output the problem. Output in this format:\nSolution:\nFinal answer: (one definite numerical value)\n\nProblem: {row['sentence']}"

            prompt.append({
                    "role": "user",
                    "content": f"Problem:\n{row['sentence']}"
                })
            #prompt = copy.deepcopy(template)
        promptsList.append(prompt)
        
        #print(prompt)
        
    logger.info(f"Genarate {len(promptsList)} pieces of prompts")
    return promptsList

def generate_TruthfulQA_Prompts(dataset:str, model:str, df:df=None, shot:int=1) -> list:
    #logger = getMyLogger('fileLogger','log.log')

    # get template from  'prompt.json'
    template = getTemplate(dataset)
    #rawContent = template[1]['content']
    
    promptsList = []
    for index, row in df.iterrows():
        #jsonFormat = list2json(row['tokens'])
    
        if model == 'gpt-3.5-turbo-instruct' or model == "meta/meta-llama-3-8b" or model == "meta/meta-llama-3-70b" or model.startswith('meta/llama-2-'):
            shot_number = shot
            examples_str = ""
            while shot_number > 0:
                random_row = df.sample(n=1)
                while len(random_row['sentence'].values[0]) > 500:
                    random_row = df.sample(n=1)
                if row['index'] == random_row['index'].values[0]:
                    continue
                if random_row['sentence'].values[0] in examples_str:
                    continue
                example_str = f"{random_row['sentence'].values[0]}\nAnswer: {random_row['answer'].values[0]}\n"
                examples_str += example_str
                shot_number-=1

            prompt = f"{template[0]['content']}\n{examples_str}\n{row['sentence']}\nAnswer:"
        else:
            shot_number = 0
            prompt = [{
                "role": "system", "content": f"{template[0]['content']}"
            }]
            while shot_number < shot:
                
                random_row = df.sample(n=1)
                while len(random_row['sentence'].values[0]) > 600:
                    random_row = df.sample(n=1)
                if row['index'] == random_row['index'].values[0]:
                    continue
                """ if random_row['sentence'].values[0] in examples_str:
                    continue """
                prompt.append({
                        "role": "user",
                        "content": f"{random_row['sentence'].values[0]}\n\nAnswer:"
                    })
                prompt.append({
                        "role": "assistant",
                        "content": f"{random_row['answer'].values[0]}"
                    })
                shot_number += 1

            prompt.append({
                        "role": "user",
                        "content": f"{row['sentence']}\n\nAnswer:"
                    })
            """template[1]['content'] = f"{examples_str}\n{row['sentence']}\nAnswer:"
            prompt = copy.deepcopy(template) """
        promptsList.append(prompt)
        
        #print(prompt)
        
    logger.info(f"Genarate {len(promptsList)} pieces of prompts")
    return promptsList

def list2json(tokenList)->str:
    result = {word: '' for word in ast.literal_eval(tokenList)}
    return result


def getTemplate(dataset:str) -> str:
    import json
    prompts = None
    try:
        with open('./prompt.json', 'r', encoding='utf-8') as f:
            prompts = json.load(f)
            template = prompts[dataset]
            logger.info(f"Get template for {dataset} successfully.")
    except Exception as e:
        logger.error(f"Somthing Error: {e}")

    return template


def loadDataset(dataset:str) -> df:
    #logger = getMyLogger('fileLogger','log.log')

    import json
    datasetPaths = None
    try:
        with open('./datasetPaths.json', 'r', encoding='utf-8') as f:
            datasetPaths = json.load(f)
            path = datasetPaths[dataset]
    except Exception as e:
        logger.error(f"Somthing Error: {e}")

    df = pd.read_csv(path, encoding='utf-8')
    logger.info(f"Load {dataset} datasets successfully. Shape: {df.shape}, columns: {df.columns.values}")
    return df

def load_dataset_from_hf(dataset:str, path:str) -> df:
    try:
        dataset_dict = load_dataset('CodeMixBench/CodeMixBench', data_files={'test': path})

    except Exception as e:
        logger.error(f"Somthing Error: {e}")

    # Get the 'test' split
    test_dataset = dataset_dict['test']

    # Convert to a Pandas DataFrame
    test_df = test_dataset.to_pandas()
    logger.info(f"Load {dataset} datasets successfully. Shape: {test_df.shape}, columns: {test_df.columns.values}")

    return test_df


async def asyncAskGPT(
        answerFilePath:str,
        messageList:list,
        indices:list,
        model:str = "gpt-3.5-turbo",
        temperature:float = 0.8,
        top_p:float = 0.95,
        api_key:str = None,
        base_url:str = None,
        max_tokens:int = 1024,
        ) -> df:
    
    if model.startswith('gpt'):
        logdir = f"oaib_gpt.txt"
    elif model.startswith('meta'):
        logdir = f"oaib_llama.txt"
    elif model.startswith('mistral'):
        logdir = f"oaib_mistral.txt"
    else:
        logdir = f"oaib.txt"

    # set rate limits.
    batch = Batch(
        rpm = 1000,
        tpm = 500_000,
        workers = 100,
        safety = 0.1,
        api_key = api_key,
        base_url = base_url,
        logdir = logdir
    )
    results = None
    
    # Creating a batch with many chat completions.
    for index in indices:
        if model == 'gpt-3.5-turbo-instruct':
            await batch.add(
                index,
                "completions.create",
                model=model,
                prompt= messageList[index],
                temperature = temperature,
                top_p = top_p,
                max_tokens = max_tokens,
                #response_format={"type": "json_object"}
            )
        else:
            await batch.add(
                index,
                "chat.completions.create",
                model=model,
                messages = messageList[index],
                temperature = temperature,
                top_p = top_p,
                max_tokens = max_tokens,
                #response_format={"type": "json_object"}
            )
    
    results = await batch.run()
    if  len(results) == 0:
        logger.warning(f"Empty Result.")
        exit()

    batch = Batch(
        rpm = 1000,
        tpm = 300_000,
        workers = 100,
        safety = 0.1,
        api_key = api_key,
        base_url = base_url,
        logdir = logdir
    )

    missingIndices = []
    """ if '_all_' in answerFilePath:
        missingIndices = checkMissingIndices(results, len(messageList)) """
    missingIndices = checkMissingIndices(results, len(messageList))
    count = 1
    while len(missingIndices):
        if count > 5:
            break
        
        logger.info(f"Try to ask index {missingIndices} in the {count} attempt")

        for index in missingIndices:
            #logger.debug(f"messageList[{errIndex}]: {messageList[errIndex]}")
            if model == 'gpt-3.5-turbo-instruct':
                await batch.add(
                    index,
                    "completions.create",
                    model=model,
                    prompt= messageList[index],
                    temperature = temperature,
                    top_p = top_p,
                    max_tokens = max_tokens,
                    #response_format={"type": "json_object"}
                )
            else:
                await batch.add(
                    index,
                    "chat.completions.create",
                    model=model,
                    messages = messageList[index],
                    temperature = temperature,
                    top_p = top_p,
                    max_tokens = max_tokens,
                    #response_format={"type": "json_object"}
                )
        againResult = await batch.run()
        if  len(againResult) == 0:
            logger.warning(f"Empty againResult.")
            #exit()
        else:
            results = pd.concat([results, againResult], ignore_index=True)
        
            results = results.sort_values(by=['index'])

        missingIndices = checkMissingIndices(results, len(messageList))

        count = count + 1

    results = results.sort_values(by=['index'])
    results.to_csv(answerFilePath, index=False, encoding='utf-8')
    
    return results

async def asyncAskReplicate(
        answerFilePath:str,
        messageList:list,
        indices:list,
        model:str = "meta/meta-llama-3-8b",
        temperature:float = 0.8,
        top_p:float = 0.95,
        api_key:str = None,
        max_tokens:int = 1024,
        ) -> df:
    if model == "meta/meta-llama-3-70b":
        max_token_key = 'max_tokens'
        stop_key = 'stop'
    else:
        max_token_key = 'max_new_tokens'
        stop_key = 'stop_sequences'
    
    # set rate limits.
    batch = BatchReplicate(
        rpm = 1000,
        tpm = 500_000,
        workers = 100,
        safety = 0.1,
        api_key = None,
        timeout= 3600
    )
    results = None
    
    # Creating a batch with many chat completions.
    for index in indices:
        input = {
            'prompt': messageList[index],
            'temperature': temperature,
            'top_p': top_p,
            max_token_key: max_tokens,
            'frequency_penalty': 0,
            'presence_penalty': 0,
            stop_key: 'stop'
        }

        await batch.add(
            index,
            "replicate.models.predictions.create",
            model=model,
            input= input,
        )

    
    results = await batch.run()
    if  len(results) == 0:
        logger.warning(f"Empty Result.")
        exit()

    batch = BatchReplicate(
        rpm = 1000,
        tpm = 300_000,
        workers = 100,
        safety = 0.1,
        api_key = None,
        timeout= 3600
    )

    missingIndices = []
    """ if '_all_' in answerFilePath:
        missingIndices = checkMissingIndices(results, len(messageList)) """
    missingIndices = checkMissingIndices(results, len(messageList))
    count = 1
    while len(missingIndices):
        if count > 5:
            break
        
        logger.info(f"Try to ask index {missingIndices} in the {count} attempt")

        for index in missingIndices:
            #logger.debug(f"messageList[{errIndex}]: {messageList[errIndex]}")
            input = {
                'prompt': messageList[index],
                'temperature': temperature,
                'top_p': top_p,
                max_token_key: max_tokens,
                'frequency_penalty': 0,
                'presence_penalty': 0,
                stop_key: 'stop'
            }

            await batch.add(
                index,
                "replicate.models.predictions.create",
                model=model,
                input= input,
            )
        againResult = await batch.run()
        if  len(againResult) == 0:
            logger.warning(f"Empty againResult.")
            #exit()
        else:
            results = pd.concat([results, againResult], ignore_index=True)
        
            results = results.sort_values(by=['index'])

        missingIndices = checkMissingIndices(results, len(messageList))

        count = count + 1

    results = results.sort_values(by=['index'])
    results.to_csv(answerFilePath, index=False, encoding='utf-8')
    
    return results


async def analyse_Token_Label_Result(
        results:df,
        answerFilePath:str,
        rawPath:str,
        predFilePath:str,
        messageList:list,
        model:str = "gpt-3.5-turbo",
        temperature:float = 0.8,
        top_p:float = 0.95,
        true_df:df = None,
        again = 3,
        api_key:str = None,
        base_url:str = None,
        ) ->df :
    
    # set rate limits.
    batch = Batch(
        rpm = 5000,
        tpm = 100_000,
        workers = 80,
        safety = 0.1,
        api_key = api_key,
        base_url = base_url
    )

    missingIndices = []
    if '_all_' in predFilePath:
        missingIndices = checkMissingIndices(results, len(messageList))
    count = 1
    while len(missingIndices):
        if count > again:
            break
        
        logger.info(f"Try to ask index {missingIndices} in the {count} attempt")

        for index in missingIndices:
            #logger.debug(f"messageList[{errIndex}]: {messageList[errIndex]}")
            if model == 'gpt-3.5-turbo-instruct':
                await batch.add(
                    index,
                    "completions.create",
                    model=model,
                    prompt= messageList[index],
                    temperature = temperature,
                    top_p = top_p,
                    max_tokens = 1024,
                    #response_format={"type": "json_object"}
                )
            else:
                await batch.add(
                    index,
                    "chat.completions.create",
                    model=model,
                    messages = messageList[index],
                    temperature = temperature,
                    top_p = top_p,
                    max_tokens = 1024,
                    #response_format={"type": "json_object"}
                )
        againResult = await batch.run()
        if  len(againResult) == 0:
            logger.warning(f"Empty againResult.")
            exit()

        results = pd.concat([results, againResult], ignore_index=True)
        
        results = results.sort_values(by=['index'])

        missingIndices = checkMissingIndices(results, len(messageList))

        count = count + 1

    pred, formatErrorIndices = result2pred(results, true_df, model)

    logger.warning(f"index {formatErrorIndices} have format errors !")
    pred['true'] = true_df['answer']
    pred.to_csv(predFilePath, index=False, encoding='utf-8')
    logger.info(f" Save the final prediction into {predFilePath}.")
    results.to_csv(rawPath, index=False, encoding='utf-8')
    logger.info(f" Save the answers into {rawPath}.")
    return pred

async def analyse_Sentence_Label_Result(
        results:df,
        answerFilePath:str,
        predFilePath:str,
        messageList:list,
        task:str,
        model:str = "gpt-3.5-turbo",
        temperature:float = 0.8,
        top_p:float = 0.95,
        true_df:df = None,
        again = 3,
        api_key:str = None,
        base_url:str = None,
        ) ->df :

    indices = []
    preList = []
    sentenceList = []
    trueList = []

    #columns = results.columns.values.tolist()


    
    missingIndices = checkMissingIndices(results, len(messageList))
    count = 1
    while len(missingIndices):
            # set rate limits.
        batch = Batch(
            rpm = 5000,
            tpm = 100_000,
            workers = 80,
            safety = 0.1,
            api_key = api_key,
            base_url = base_url
        )

        if count > again:
            break
        
        logger.info(f"Try to ask index {missingIndices} in the {count} attempt")

        for index in missingIndices:
            #logger.debug(f"messageList[{errIndex}]: {messageList[errIndex]}")
            if model == 'gpt-3.5-turbo-instruct':
                await batch.add(
                    index,
                    "completions.create",
                    model=model,
                    prompt= messageList[index],
                    temperature = temperature,
                    top_p = top_p,
                    max_tokens = 1024,
                    #response_format={"type": "json_object"}
                )
            else:
                await batch.add(
                    index,
                    "chat.completions.create",
                    model=model,
                    messages = messageList[index],
                    temperature = temperature,
                    top_p = top_p,
                    max_tokens = 1024,
                    #response_format={"type": "json_object"}
                )
        againResult = await batch.run()
        if  len(againResult) == 0:
            logger.warning(f"Empty againResult.")
            exit()

        results = pd.concat([results, againResult], ignore_index=True)
        
        results = results.sort_values(by=['index'])
        results.to_csv(answerFilePath, index=False, encoding='utf-8')
        missingIndices = checkMissingIndices(results, len(messageList))

        count = count + 1

    answer_list = []
    # Extract the indices and the predicted value list from result and form a new df.
    for i, row in results.iterrows():
        index = row['index']
        #logger.debug(f"row['result']: {row['result']}")
        # turn str to dict, for get the answer from gpt
        try:
            result_dict = ast.literal_eval(str(row['result']))

        except Exception as e:
            logger.error(f"Somthing Error: {e}:\n index{index}: {str(row['result'])}")
            exit()
        
        try:
            sentence = true_df.loc[true_df.loc[results['index'] == index].index[0], 'sentence']
            true_tag = true_df.loc[true_df.loc[results['index'] == index].index[0], 'answer']
        except:
            logger.error(f"Fail to get sentence and true label from true_df: {index}")
            raise

        if model == 'gpt-3.5-turbo-instruct':
            answer = result_dict['choices'][0]['text']
        elif model == 'meta/meta-llama-3-8b' or model == 'meta/meta-llama-3-70b' or model.startswith('meta/llama-2-'):
            answer = result_dict['output']
        else:
            answer = result_dict['choices'][0]['message']['content']
        #logger.debug(f"answer: {answer}")
        try:
            pred = answer.strip()
        except:
            logger.error(f"Fail to strip answer: {index}\nanswer")
            pred = ''

        if task == 'mmlu':
            #pred = pred.replace('(С)', '(C)').replace('(ए)', '(A)').replace('(ब)', '(B)').replace('(सी)', '(C)').replace('(ड)', '(D)').replace('(अ)', '(A)').replace('(प)', '(A)').replace('(स)', '(C)').replace('(ग)', '(C)').replace('(डी)', '(D)').replace('(घ)', '(D)')
            pattern = r"[ABCD]"
            pred_re = re.findall(pattern, answer, re.MULTILINE)
            if len(pred_re) > 0:
                pred = pred_re[0].strip()
            else:
                pred = 'unk'
                logger.warning(f"index {index} has error, answer: {answer}")
        
        if task =='gsm8k':

            #answers = answer.split('\n')
            if "Final Answer:" in answer:
                answer_ = answer.split('Final Answer:', 1)[1]
            elif "Final answer:" in answer:
                answer_ = answer.split('Final answer:', 1)[1]
            else:
                answer_ = answer
                logger.warning(f"index {index} has format error, answer:\n{answer}")

            if '[stop]' in answer:
                answer_ = answer_.split('[stop]', 1)[0]

            pattern = r"\-?[0-9\.\,]+"
            pred_re = re.findall(pattern, answer_.strip('.'), re.MULTILINE)
            pred_re = [x.strip().strip('.').replace(',','').replace(':','') for x in pred_re if x != '.' and x != ',']
            if len(pred_re) == 1:
                pred = pred_re[0].strip().strip('.').replace(',','').replace(':','')
            elif len(pred_re) == 0:
                pred = 'unk'
                logger.warning(f"index {index} has error, answer:\n{answer}")
            else:
                if str(true_tag) in pred_re:
                    pred = true_tag
                else:
                    pred = pred_re[0]
                    logger.warning(f"index {index} has error, answer:\n{answer}")
            #match = re.search(pattern, answer, re.MULTILINE)
        
        if task == 'truthfulqa':
            """ if '\n' in answer:
                lines = answer.split('\n')
                pred = lines[-1] """
            pattern = r"[ABCDEFGHIJKLMN]"
            pred_re = re.findall(pattern, answer, re.MULTILINE)
            if len(pred_re) > 0:
                pred = pred_re[0].strip()
            else:
                pred = 'unk'
                logger.warning(f"index {index} has error, answer: {answer}")
        

        indices.append(index)
        preList.append(pred)
        trueList.append(true_tag)
        sentenceList.append(sentence)
        answer_list.append(answer)

    if len(indices) == len(preList) == len(sentenceList):
        pred_df = pd.DataFrame({
            'index': indices,
            'pred': preList,
            'true': trueList,
            'answer': answer_list,
            'sentence': sentenceList
        })
        sorted_pred = pred_df.sort_values(by=['index'])
        
        logger.info(f"Extract {len(sorted_pred)} pieces of predictions from GPT's answer result (sentence-level task).")
    else: 
        logger.warning(f"length of indices !== length of preList")
        exit()
    
    sorted_pred.to_csv(predFilePath, index=False, encoding='utf-8')
    logger.info(f" Save the final prediction into {predFilePath}.")

    return sorted_pred

def loadAnswerFile(answerFilePath:str) -> df:

    try:
        results = pd.read_csv(answerFilePath)
        logger.info(f"Get {len(results)} pieces of answers from {answerFilePath}.")
    except Exception as e:
        logger.error(f"Somthing Error: {e}")

    return results


def checkMissingIndices(results:df, dataset_len):
    indices = results['index'].tolist()
    if dataset_len == len(indices):
        return []
    missingIndices = list(set(range(dataset_len)) - set(indices))

    return missingIndices

def labelMissingToken() -> None:
    

    return

def result2pred(results:df, true_df:df, model:str):
    
    result_df = results
    true_df = true_df

    indices = []
    preList = []
    tokensList = []
    errList = []

    # Extract the indices and the predicted value list from result and form a new df.
    for i, row in result_df.iterrows():
        index = row['index']
        true_tokens = eval(str(true_df.loc[true_df['index'] == index, 'tokens'].iloc[0]))
        
        #logger.debug(f"row['result']: {row['result']}")
        # turn str to dict, for get the answer from gpt
        try:
            result_dict = ast.literal_eval(str(row['result']))

        except Exception as e:
            logger.error(f"Somthing Error: {e}:\n index{index}: {str(row['result'])}")
            errList.append(int(index))

        if model == 'gpt-3.5-turbo-instruct':
            answer = result_dict['choices'][0]['text']
            pattern = r"\[.*?\]"
            re_result = re.findall(pattern, answer)
            if re_result:
                answer = re_result[0]

        else:
            answer = result_dict['choices'][0]['message']['content']
        #logger.debug(f"answer: {answer}")

        # answer is in a list of jsons format, get the prdicted value of each element
        pred, tokens = getPredictedValue(answer, index)
        
        
        if len(pred) == 0:
            errList.append(int(index))
            preList.append(['unk']*len(true_tokens))
        else:
            pred_dict = dict(zip(tokens, pred))
            pred_lid = []
            for token in true_tokens:
                if token in tokens:
                    pred_lid.append(pred_dict[token])
                else:
                    pred_lid.append('unk')
            preList.append(pred_lid)

        indices.append(index)
        tokensList.append(true_tokens)

    if len(indices) == len(preList) == len(tokensList):
        pred_df = pd.DataFrame({
            'index': indices,
            'tokens': tokensList,
            'pred': preList,
        })
        sorted_pred = pred_df.sort_values(by=['index'])
        
        logger.info(f"Extract {len(sorted_pred)} pieces of predictions from GPT's answer result, indices {errList} have format errors.")
    else: 
        logger.warning(f"length of indices !== length of preList")
        exit()

    return sorted_pred, errList


def getPredictedValue(answer:str, index):
    

    """ ast.literal_eval get less error than json.loads
    json_data = json.loads(str(answer))
    pred = list(json_data.values())
    tokens = list(json_data.keys()) 
    """

    # expecting answer in format like: "[{'Wahrscheinlich': 'D'}, {'unsere': 'D'}]"
    try:
        dictList = ast.literal_eval(str(answer))
        tokens = [list(dict.keys())[0] for dict in dictList]
        pred = [list(dict.values())[0] for dict in dictList]
    except:
        try:
            answer = cleanAnswer(answer)
            dictList = ast.literal_eval(str(answer))
            tokens = [list(dict.keys())[0] for dict in dictList]
            pred = [list(dict.values())[0] for dict in dictList]
        except:
            logger.error(f"Somthing Error:\n index {index}: {str(answer)}")
        
            return [], []

    """
      if len(pred)>=5:
        logger.debug(pred[:5])
    elif len(pred)>=1:
        logger.debug(pred[:1])  
    """
    if len(tokens) == len(pred):
        return pred, tokens
    else:
        logger.error(f"len(tokens) != len(pred) index {index}:\n{tokens}\n{pred}")
        return [], []


def cleanAnswer(answer:str)->str:
    #logger.debug(f"answer before cleaning: {answer}")
    answer = answer.replace('""','"')
    answer = answer.replace('\n','')
    #logger.debug(f"answer after cleaning: {answer}")
    return answer.strip(' ,.')


def checkLength(true:df, pred:df, format_error_list:list) -> list:

    #Check the length of ture and pred for each sentence 
    checkLength = []
    lenNotEqual = []
    #messageList = []
    for i, predRow in pred.iterrows():
        index = predRow['index']
        if index in format_error_list:
            continue
        if index in true['index'].values:
            trueanswer = true.loc[true['index'] == index, 'answer'].iloc[0]
            trueTokens = true.loc[true['index'] == index, 'tokens'].iloc[0]
            
            try:
                trueList = ast.literal_eval(str(trueanswer))
            except Exception as e:
                logger.error(f"Somthing Error: {e}:\n{str(trueList)}")
            try:
                predList = ast.literal_eval(str(predRow['pred']))
                #logger.debug(f"trueList: {trueList}, predList: {predList}, {len(predList) == len(trueList)}")
            except Exception as e:
                logger.error(f"Somthing Error: {e}:\n{str(predRow)}")

            if len(predList) == len(trueList):
                checkLength.append(len(predList) == len(trueList))
                #logger.debug(f"Index {index} : Lengths of true answer list and pred answer list are {len(predList)}.")
            else:
                checkLength.append(len(predList) == len(trueList))
                lenNotEqual.append(index)
                #messageList.append(predRow['messages'])
                logger.debug(f"index {index} : Length of true answer list is {len(trueList)} while length of pred answer list is {len(predList)}.\
                             \n{trueTokens}\
                             \n{predRow['tokens']}")
    checkLength = all(checkLength)

    if not checkLength:
        logger.warning(f"The lengths of ture and pred for santences {lenNotEqual} are not equal.")
    else:
        logger.info(f"Completely check the length of ture and pred for each sentence.")

    return lenNotEqual


def compute_token_label_Metric(pred_df:df, matric_path) -> None :

    true = pred_df['true'].tolist()
    pred = pred_df['pred'].tolist()

    if not len(true) == len(pred):
        raise

    accuracy_list = []
    precision_list = []
    recall_list = []
    micro_f1_list = []
    macro_f1_list = []
    weighted_f1_list = []

    for index in range(len(pred_df)):
        
        try:
            true_label = eval(str(true[index]))
            pred_label = eval(str(pred[index]))
        except Exception:
            logger.error(f"index:{index}\ntrue[index]: {true[index]}\npred[index]: {pred[index]}\nException:{Exception}")
            exit()

        if not len(true_label) == len(pred_label):
            logger.error(f"Index:{index} lengths of true_label and pred_label are not equal.\n{true_label}\n{pred_label}")
            exit()

        try:
            accuracy_list.append(accuracy_score(true_label, pred_label)*100)
        except:
            logger.error(f"Index:{index} something error.\n{true_label}\n{pred_label}")
        #precision_list.append(precision_score(true_label, pred_label, average='macro')*100)
        #recall_list.append(recall_score(true_label, pred_label, average='macro')*100)
        micro_f1_list.append(f1_score(true_label, pred_label, average='micro')*100)
        macro_f1_list.append(f1_score(true_label, pred_label, average='macro')*100)
        weighted_f1_list.append(f1_score(true_label, pred_label, average='weighted')*100)

    pred_df['Accuracy'] = accuracy_list
    #pred_df['Precision'] = precision_list
    #pred_df['Recall'] = recall_list
    pred_df['Micro F1'] = micro_f1_list
    pred_df['Macro F1'] = macro_f1_list
    pred_df['Weighted F1'] = weighted_f1_list
    
    logger.info(f"Accuracy: {sum(accuracy_list) / len(accuracy_list)}")
    #logger.info(f"Precision: {sum(precision_list) / len(precision_list)}")
    #logger.info(f"Recall: {sum(recall_list) / len(recall_list)}")
    logger.info(f"Micro F1 Score: {sum(micro_f1_list) / len(micro_f1_list)}")
    logger.info(f"Macro F1 Score: {sum(macro_f1_list) / len(macro_f1_list)}")
    logger.info(f"Weighted F1 Score: {sum(weighted_f1_list) / len(weighted_f1_list)}")
    
    with open(matric_path, 'w', encoding='utf-8') as f:
        f.writelines(f"Accuracy: {sum(accuracy_list) / len(accuracy_list)}")
        f.writelines(f"Micro F1 Score: {sum(micro_f1_list) / len(micro_f1_list)}")
        f.writelines(f"Macro F1 Score: {sum(macro_f1_list) / len(macro_f1_list)}")
        f.writelines(f"Weighted F1 Score: {sum(weighted_f1_list) / len(weighted_f1_list)}")


    return pred_df


def compute_sentence_label_Metric(pred_df:df, matric_path) -> None :

    trueList = pred_df['true'].tolist()
    predList = pred_df['pred'].tolist()
        
    accuracy = accuracy_score(trueList, predList)
    #precision = precision_score(trueList, predList, average='macro'))
    #recall = recall_score(trueList, predList, average='macro'))
    micro = f1_score(trueList, predList, average='micro')
    macro = f1_score(trueList, predList, average='macro')
    weighted = f1_score(trueList, predList, average='weighted')
    
    logger.info(f"Accuracy: {accuracy*100}")
    #logger.info(f"Precision: {precision}")
    #logger.info(f"Recall: {sum(recall_list) / len(recall_list)}")
    logger.info(f"Micro F1 Score: {micro*100}")
    logger.info(f"Macro F1 Score: {macro*100}")
    logger.info(f"Weighted F1 Score: {weighted*100}")
    
    with open(matric_path, 'w', encoding='utf-8') as f:
        f.writelines(f"Accuracy: {accuracy*100}")
        f.writelines('\n')
        f.writelines(f"Micro F1 Score: {micro*100}")
        f.writelines('\n')
        f.writelines(f"Macro F1 Score: {macro*100}")
        f.writelines('\n')
        f.writelines(f"Weighted F1 Score: {weighted*100}")

    return 

def compute_BLEU(pred_df:df, expid:str, matric_path) -> None :
    '''
    bleu = BLEU()
    refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
        ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.']]
    sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
    bleu = bleu.corpus_score(sys, refs)
    print(bleu)
    '''

    refList = pred_df['true'].tolist()
    predList = pred_df['pred'].tolist()
    
    if '_chi_' in expid or '_man_' in expid:
        bleu = BLEU(tokenize='zh')
    elif '_eng_' in expid:
        bleu = BLEU()
    else:
        bleu = BLEU()
        
    refs = [refList]
    sys = predList

    bleu = bleu.corpus_score(sys, refs)
    logger.info(f"SacreBLEU Score: {bleu}")
    
    with open(matric_path, 'w', encoding='utf-8') as f:
        f.writelines(f"SacreBLEU Score: {bleu}")
    
    return 