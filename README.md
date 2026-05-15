# Anupama1395-CodeMixBench-Extension
Team Members: Anumpama Singh, Srivalli Kakumani, Carson Green

To run the model all you need to do is install the requirements.txt 
and run the python test_model.py
After after that, you can run our extension scripts by running the scripts in the addition folder

Results can be found in the results folder and on the paper.

<img width="496" height="334" alt="image" src="https://github.com/user-attachments/assets/edab027e-7fe0-460c-af87-7fa6729540eb" />

<img width="675" height="183" alt="image" src="https://github.com/user-attachments/assets/16eb66e1-5129-4233-bea5-fed42df52c28" />
<img width="675" height="183" alt="image" src="https://github.com/user-attachments/assets/29d9bf0c-90f9-46a5-ba04-4c0d2547075b" />
<img width="1418" height="548" alt="image" src="https://github.com/user-attachments/assets/74f445f5-6e24-4cd1-886a-8fdf1ec2aa49" />
<img width="840" height="525" alt="image" src="https://github.com/user-attachments/assets/5cdc8566-1dda-4c98-8711-28409bc64e15" />

# [EMNLP'25] CodeMixBench: Evaluating Code-Mixing Capabilities of LLMs Across 18 Languages

<p align="center">
     <a href="https://huggingface.co/datasets/CodeMixBench/CodeMixBench" target="_blank">
      <img alt="Huggingface" src="https://img.shields.io/badge/🤗-Huggingface-blue" />
   </a>
        
  <a href="https://aclanthology.org/2025.emnlp-main.109/" target="_blank">
      <img alt="Paper" src="https://img.shields.io/badge/📜-Paper-purple" />
   </a>
  <a href="https://2025.emnlp.org/" target="_blank">
      <img alt="EMNLP 2025" src="https://img.shields.io/badge/Proceedings-EMNLP2025-blue" />
   </a>

</p>

## ℹ️ Overview
Code-mixing is a linguistic phenomenon where multilingual speakers switch or mix two or more languages within a single utterance or conversation. 
To evaluate LLMs’ comprehension of multilingual code-mixed texts, we introduce CodeMixBench, a benchmark comprising eight tasks across 18 languages. 

Our benchmark comprises synthesized datasets targeting knowledge reasoning, 
mathematical reasoning, and truthfulness tasks, along with LID, POS, NER, SA, and MT tasks, 
which have been adapted from open-source studies. 
For knowledge reasoning, we developed the code-mixed MMLU (CM-MMLU) based on the MMLU test set, 
featuring multiple-choice questions from 57 subjects to assess the model's comprehensive knowledge reasoning abilities. 
For mathematical reasoning, we created the code-mixed GSM8K (CM-GSM8K), derived from the GSM8K test set, 
which evaluates mathematical reasoning capabilities with each question including step-by-step solutions. 
For truthfulness assessment, we constructed the code-mixed TruthfulQA (CM-TruthfulQA) using 817 multiple-choice 
questions from the TruthfulQA test set. 

The datasets encompass 12 languages from six families: Germanic
(en, de, nl), Romance (es, fr), Sino-Tibetan (zh),
Afro-Asiatic (ar), Indo-Aryan (hi, bn, mr, ne), and
Dravidian (ta).
We synthesized 11 code-mixed language pairs for CM-
MMLU(based on the MMLU) with 12,156 question-option-answer combinations, 4 pairs for CM-TruthfulQA with 3,122
multiple-choice instances, 4 pairs for CM-GSM8K
with 4,367 math problems, and 3 pairs for MT with
2,711 code-mixed sentences.

![Statistics of 18 languages](pics/18_languages.png)

We evaluate three families of LLMs on
CodeMixBench, revealing consistent underperformance across all models on code-mixing
datasets involving language pairs from different language families. However, enhancements
in training data size, model scale, post-training,
and few-shot learning can improve LLM performance on code-mixing datasets.
![Main evaluation results on CodeMixBench](pics/main_result.png)


## ⚙️ Setup
1. Follow these steps to set up your development environment:
   ```
   git clone git@github.com:Jeromeyluck/CodeMixBench.git
   cd CodeMixBench

   conda create -n CodeMixBench python=3.9
   conda activate CodeMixBench
   pip install -r requirements.txt
   ```
   
## 🚀 Launch CodeMixBench

   ```
   python ./test_model.py \
     --dataset lid_guaspa \
     --expid lid_guaspa_all_0shot \
     --model gpt-3.5-turbo \
     --shot 5 \
     --api sk-********************* \
     --url https://****************
   ```
   - `dataset`: select the dataset (e.g., `lid_gereng`, `lid_spaeng`, `ner_hineng`).
   - `expid`: define the ID of the test, the results file will be named after this ID.
   - `model`: the model you test. The default model is `gpt-3.5-turbo`.
   - `shot`: use for few-shot test (by default it will be `1`).
   - `api`: API Key (default key will be `OPENAI_API_KEY` defined in system path).
   - `url`: API function provider's URL.



## 📑 Cite US

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

  ```
@inproceedings{yang-chai-2025-codemixbench,
    title = "{C}ode{M}ix{B}ench: Evaluating Code-Mixing Capabilities of {LLM}s Across 18 Languages",
    author = "Yang, Yilun  and
      Chai, Yekun",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.109/",
    doi = "10.18653/v1/2025.emnlp-main.109",
    pages = "2139--2169",
    ISBN = "979-8-89176-332-6",
    abstract = "Code-mixing, the practice of switching between languages within a conversation, poses unique challenges for traditional NLP. Existing benchmarks like LinCE and GLUECoS are limited by their narrow language pairs and tasks, failing to adequately assess large language models' (LLMs) code-mixing abilities. Despite the recognized importance of code-mixing for multilingual users, research on LLMs in this context remains sparse. Additionally, current techniques for synthesizing code-mixed data are underdeveloped to generate code-mixing. In response, we introduce CodeMixBench, a comprehensive benchmark covering eight tasks, including three specific to LLMs and five traditional NLP tasks, and 18 languages from seven language families. We also propose a new method for generating large-scale synthetic code-mixed texts by combining word substitution with GPT-4 prompting. Our evaluation reveals consistent underperformance of LLMs on code-mixed datasets involving different language families. Enhancements in training data size, model scale, and few-shot learning could improve their performance. The code and dataset are available at https://github.com/Jeromeyluck/CodeMixBench."
}
  ```

