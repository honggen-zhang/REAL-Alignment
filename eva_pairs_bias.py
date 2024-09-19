import datasets
import torch
import random
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from utils import (
    get_local_dir,TemporarilySeededRandom
)

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

eval_set = datasets.load_dataset('honggen/pythia-harm', split='train',data_dir = 'random')
i = 0
chosen_example = []
rejected_example = []
for ex in eval_set:
    # generate here is a placeholder for your models generations
    print('====================')
    prompt = extract_anthropic_prompt(ex['chosen'])
    chosen_response = ex['chosen'][len(prompt):]

    rejected_response = ex['rejected'][len(prompt):]
    example = {}
    example["instruction"] = prompt
    example["output"] = chosen_response
    example["generator"] = 'chosen'
    chosen_example.append(example)
    example1 = {}
    example1["instruction"] = prompt
    example1["output"] = rejected_response
    example1["generator"] = 'rejected'
    rejected_example.append(example1)
    
random.seed(28)
rejected_example = random.sample(rejected_example,200)
random.seed(28)
chosen_example = random.sample(chosen_example,200)

import json

# Define the filename
filename = './example/bias/pythia_chosen_random_harm.json'

# Open a file in write mode and save the list of dictionaries
with open(filename, 'w') as f:
    json.dump(chosen_example, f, indent=4)
    
# Define the filename
filename = './example/bias/pythia_rejected_random_harm.json'

# Open a file in write mode and save the list of dictionaries
with open(filename, 'w') as f:
    json.dump(rejected_example, f, indent=4)