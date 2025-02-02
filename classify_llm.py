#!/usr/bin/env python
# On 2025/02/01 written by ChatGPT 4o, modified by Yuen-Hsien Tseng and ChatGPT o3-mini-high
# conda create -n deepseek python=3.11
# conda activate deepseek
# pip install ollama scikit-learn openai python-dotenv
# You need to prepare a '.env' file and store the OPENAI_API_KEY there. 
# See the bottom of this file to know how to run this script.
"""
A modular command-line friendly program for text classification using few-shot prompts.
It supports both Ollama and OpenAI backends. The program reads training and test files
(with two tab-separated columns: label and text) from a given dataset folder.
Usage:
    python classify.py <model> <dataset_folder> <num_shots> <prompt_version> [--backend <ollama|openai>] [--datasets <dataset1 dataset2 ...>]
Example:
    python classify.py gemma:2b ../TxtClf_Dataset 5 1 --backend ollama
"""

import argparse
import time, os, random
import collections
from sklearn.metrics import f1_score, confusion_matrix

from dotenv import load_dotenv
load_dotenv()

# Try importing both Ollama and OpenAI SDKs.
try:
    import ollama
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    pass


def read_data(file_path):
    """
    Reads a tab-separated file and returns lists of texts and labels.
    """
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            label, text = line.strip().split('\t')
            labels.append(label)
            texts.append(text)
    return texts, labels


def get_few_shot_examples(train_texts, train_labels, num_examples=5):
    """
    Returns a string containing few-shot examples (sampled in a stratified way)
    and also returns the list of label strings.
    """
    label_to_texts = collections.defaultdict(list)
    for text, label in zip(train_texts, train_labels):
        label_to_texts[label].append(text)
    
    num_classes = len(label_to_texts)
    # Here we choose "num_examples" per class.
    examples_per_class = num_examples
    print("Number of classes:", num_classes, ", examples per class:", examples_per_class)
    label_list = list(label_to_texts.keys())
    print("Labels:", label_list)
    
    examples = []
    # Use a fixed random seed so that results are reproducible.
    random.seed(42)
    for label, texts in label_to_texts.items():
        sampled_texts = random.sample(texts, min(examples_per_class, len(texts)))
        for text in sampled_texts:
            examples.append(f"Text: {text}\nLabel: {label}")
    
    random.shuffle(examples)  # Shuffle to avoid ordering bias.
    return "\n\n".join(examples), label_list


def reverse_string(s):
    """
    Returns the reverse of the input string.
    """
    return s[::-1]

def extract_label(predicted_output, label_list):
    """
    Extracts a label from the predicted output by searching for the known label strings.
    This is done by reversing the string and finding the first match.
    """
    min_positions = [reverse_string(predicted_output).find(reverse_string(label)) for label in label_list]
    # Replace -1 (not found) with a large number.
    min_positions = [len(predicted_output) if p == -1 else p for p in min_positions]
    idx = min_positions.index(min(min_positions))
    return label_list[idx]


def send_request(prompt, backend, model):
    """
    Sends the prompt to the specified backend (ollama or openai) using the given model,
    and returns the response text.
    """
    if backend == "ollama":
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content'].strip()
    elif backend == "openai":
        # Make sure that your OpenAI API key is set (e.g., via openai.api_key or environment variable).
        # OpenAI configuration
        client = OpenAI( api_key=os.getenv('OPENAI_API_KEY') )        
        response = client.chat.completions.create(
            model=model, # model="gpt-4o"
            messages=[
                {"role": "system", "content": "You are a highly accurate text classification model."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    else:
        raise ValueError("Unsupported backend. Choose either 'ollama' or 'openai'.")


def classify_text(text, label_list, few_shot_examples, backend, model, prompt_version):
    """
    Constructs a few-shot prompt (using one of two versions) and sends it to the chosen backend.
    Returns the predicted label.
    """
    # Prompt version 1.
    prompt_1 = (
        "You are a highly accurate text classification model. "
        "Your task is to assign the correct category label to a given text.\n"
        f"The label strings represent distinct topics and are in the list: {label_list}\n"
        "Your response should only return one of the label strings in the above list and nothing else.\n\n"
    )
    if few_shot_examples:
        prompt_1 += (
            "Here are some examples:\n\n"
            f"{few_shot_examples}\n\n"
            "Now, try your best based on the labels and the above examples to classify the following text:\n"
        )
    else:
        prompt_1 += (
            "Now, try your best based on the labels to classify the following text:\n"
        )
    prompt_1 += f"Text: {text}\nLabel:"

    # Prompt version 2.
    prompt_2 = (
        "You are a highly accurate text classification expert. \n"
        "Your task is to assign the correct category label to a given text. \n"
        f"The label strings represent distinct topics and are in the list: {label_list}\n"
        "The label strings should have some semantic meaning for you to classify the text.\n"
        "You could also infer the category's semantic meaning from the below examples, if any.\n\n"
        "Here are some examples, if any:\n\n"
        f"{few_shot_examples}\n\n"
        "Now, try your best based on the above examples and the label string meanings to classify the following text:\n\n"
        f"Text: {text}\nLabel:"
    )
    
    prompt_list = [prompt_1, prompt_2]
    prompt = prompt_list[prompt_version - 1]
    
    predicted_output = send_request(prompt, backend, model)
    predicted_label = extract_label(predicted_output, label_list)
    
    # Debug prints: show the prompt, predicted label, and full output.
    print("==== Prompt ====")
    print(prompt)
    print("==== Predicted Label ====")
    print(predicted_label)
    print("==== Full Output ====")
    print(predicted_output)
    print()
    
    return predicted_label

def process_dataset(dataset, dataset_folder, num_shots, prompt_version, backend, model):
    """
    Processes one dataset: reads training and test files, builds few-shot examples, classifies
    each test text, computes metrics, and prints the results.
    """
    # Determine file paths (special case for CTC).
    if dataset == "CTC":
        train_file = f"{dataset_folder}/{dataset}_train_sl.txt"
        test_file = f"{dataset_folder}/{dataset}_test_sl.txt"
    else:
        train_file = f"{dataset_folder}/{dataset}_train.txt"
        test_file = f"{dataset_folder}/{dataset}_test.txt"
    
    print(f"Processing dataset: {dataset}")
    train_texts, train_labels = read_data(train_file)
    test_texts, test_labels = read_data(test_file)
    
    # Build few-shot examples from training data.
    few_shot_examples, label_list = get_few_shot_examples(train_texts, train_labels, num_examples=num_shots)
    
    # Classify each test text.
    predicted_labels = []
    start_time = time.time()
    for i, text in enumerate(test_texts):
        print(f"\nClassifying test item {i+1}/{len(test_texts)}:")
        pred_label = classify_text(text, label_list, few_shot_examples, backend, model, prompt_version)
        predicted_labels.append(pred_label)
    elapsed_time = time.time() - start_time
    
    # Compute evaluation metrics.
    micro_f1 = f1_score(test_labels, predicted_labels, average="micro", zero_division=0)
    macro_f1 = f1_score(test_labels, predicted_labels, average="macro", zero_division=0)
    cm = confusion_matrix(test_labels, predicted_labels)
    
    print("Confusion Matrix:")
    print(cm)
    print("llm_model\tDataset\tMicroF1\tMacroF1\tTime(s)\tnum_shots\tprompt_version")
    print(f"{model}\t{dataset}\t{micro_f1:.4f}\t{macro_f1:.4f}\t{elapsed_time:.2f}\t{num_shots}\t{prompt_version}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Text classification using few-shot examples with either Ollama or OpenAI backend."
    )
    parser.add_argument("model", type=str, help="LLM model name (e.g., gemma:2b for Ollama or an OpenAI model name).")
    parser.add_argument("dataset_folder", type=str, help="Folder containing the datasets.")
    parser.add_argument("num_shots", type=int, help="Number of few-shot examples per class (or total, depending on implementation).")
    parser.add_argument("prompt_version", type=int, choices=[1, 2], help="Prompt version to use (1 or 2).")
    parser.add_argument("--backend", type=str, choices=["ollama", "openai"], default="ollama",
                        help="Backend to use: 'ollama' or 'openai'. Default is 'ollama'.")
    parser.add_argument("--datasets", nargs="+", default=['CnonC', 'PCNews', 'PCWeb', 'Reuters', 'CTC'],
                        help="List of dataset names to process. Default: CnonC PCNews PCWeb Reuters CTC.")
    args = parser.parse_args()
    
    print(f"Model: {args.model}, Dataset folder: {args.dataset_folder}, Num shots: {args.num_shots}, Prompt version: {args.prompt_version}, Backend: {args.backend}")
#    for dataset in args.datasets:
    for dataset in ['CnonC']:
        process_dataset(dataset, args.dataset_folder, args.num_shots, args.prompt_version, args.backend, args.model)

if __name__ == "__main__":
    main()

'''
On 2025/02/01
(deepseek) sam@ai4-glis:~/Chinese_Skewed_TxtClf$ nohup python classify_llm.py deepseek-r1:32b Datasets 3 1
Confusion Matrix:
[[49  1]
 [ 8 42]]
llm_model	Dataset	MicroF1	MacroF1	Time	num_shots	prompt
deepseek-r1:32b	CnonC	0.9100	0.9096	1433.03	3	1
deepseek-r1:32b	CnonC	0.9000	0.8990	1218.10	0	1


2025/02/01 執行：
(deepseek) sam@ai4-glis:~/Chinese_Skewed_TxtClf$ python classify_llm.py gpt-4o  Datasets 0 1 --backend openai
Confusion Matrix:
[[50  0]
 [ 8 42]]
llm_model       Dataset MicroF1 MacroF1 Time(s) num_shots   prompt_version
gpt-4o          CnonC   0.9200  0.9195  138.11  0       1
2025/02/02 每次執行大語言模型，以都會不一樣，但效果變化沒有那麼大：
(deepseek) sam@ai4-glis:~/Chinese_Skewed_TxtClf$ python classify_llm.py gpt-4o  Datasets 0 1 --backend openai
Confusion Matrix:
[[49  1]
 [ 9 41]]
llm_model	    Dataset	MicroF1	MacroF1	Time(s)	num_shots	prompt_version
gpt-4o	        CnonC	0.9000	0.8994	113.09	0	1

2025/02/01 執行：
(deepseek) sam@ai4-glis:~/Chinese_Skewed_TxtClf$ python classify_llm.py o1  Datasets 0 1 --backend openai
Confusion Matrix:
[[50  0]
 [15 35]]
llm_model       Dataset MicroF1 MacroF1 Time(s) num_shots    prompt_version
o1              CnonC   0.8500  0.8465  437.94  0   1
2025/02/02 每次執行大語言模型，以都會不一樣，但效果變化沒有那麼大：
(deepseek) sam@ai4-glis:~/Chinese_Skewed_TxtClf$ python classify_llm.py o1  Datasets 0 1 --backend openai
Confusion Matrix:
[[50  0]
 [13 37]]
llm_model	    Dataset	MicroF1	MacroF1	Time(s)	num_shots	prompt_version
o1	            CnonC	0.8700	0.8678	379.12	0	1


(deepseek) sam@ai4-glis:~/Chinese_Skewed_TxtClf$ python classify_llm.py gemma:2b Datasets 3 1 --backend ollama
gemma:2b	CnonC	0.5100	0.3856	16.18	0	1 # 2025/02/01 執行
gemma:2b	CnonC	0.6200	0.5559	14.54	2	1 # 2025/02/01 執行
gemma:2b	CnonC	0.5900	0.5524	16.85	3	1 # 2025/02/01 執行
gemma:2b	CnonC	0.6200	0.5824	17.23	3	1 # 2025/02/02 執行


每次執行較小的模型，效果會比較不穩定：
(deepseek) sam@ai4-glis:~/Chinese_Skewed_TxtClf$ python classify_llm.py mistral Datasets 0 1 --backend ollama
mistral	    CnonC	0.7800	0.7786	14.43	0	1 # 2025/02/01 執行
mistral	    CnonC	0.9100	0.9096	16.79	3	1 # 2025/02/01 執行
(deepseek) sam@ai4-glis:~/Chinese_Skewed_TxtClf$ python classify_llm.py mistral Datasets 0 1 --backend ollama
Confusion Matrix:
[[46  4]
 [11 39]]
llm_model	Dataset	MicroF1	MacroF1	Time(s)	num_shots	prompt_version
mistral	    CnonC	0.8500	0.8493	16.17	0	1 # 2025/02/02 執行
mistral	    CnonC	0.9200	0.9197	14.38	3	1 # 2025/02/02 執行
'''
