import datasets
import torch
import torch.nn as nn
import json
import random
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_local_dir

def find_last_punctuation(s):
    # Find the last occurrence of '.', '?', and '!'
    last_period = s.rfind('.')
    last_question = s.rfind('?')
    last_exclamation = s.rfind('!')

    # Get the maximum of the indices (the most recent punctuation mark)
    return max(last_period, last_question, last_exclamation)

def extract_anthropic_prompt(prompt_and_response):
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

def extract_chosen(prompt_and_response):
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[search_term_idx + len(search_term):]


def generator_response(model_name, model, tokenizer, prompts, max_new_tokens=100, temperature=0.7, top_k=30, top_p=0.95, early_stopping=True):
    batch_size = 1  # Adjust batch size according to your GPU/CPU memory limit
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    all_examples = []

    for i in tqdm.tqdm(range(num_batches)):
        batch_prompts = prompts[i * batch_size:(i + 1) * batch_size]
        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to('cuda')

        outputs = model.module.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            #temperature=temperature,
            #do_sample=True,
            #top_k=top_k,
            #top_p=top_p,
            #eos_token_id=tokenizer.eos_token_id,
            #pad_token_id=tokenizer.eos_token_id
        )

        for idx, output in enumerate(outputs):
            decoded_output = tokenizer.decode(output, skip_special_tokens=True)
            prompt_length = len(tokenizer.decode(inputs['input_ids'][idx], skip_special_tokens=True))
            #print(decoded_output)
            decoded_output = decoded_output[prompt_length:]
            #end_index = find_last_punctuation(decoded_output)
            #print(decoded_output)
            all_examples.append({
                'instruction': batch_prompts[idx],
                'output': decoded_output,
                'generator': f"{model_name}"
            })

    return all_examples

model_name_list = ['easy','hard','random','centroid','sft']
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1.5",token='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',cache_dir=get_local_dir('./cache'))
#model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1.5",token='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',cache_dir=get_local_dir('./cache'))
model = nn.DataParallel(model)  # Utilize multiple GPUs
model = model.to('cuda')
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1.5", token='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', cache_dir=get_local_dir('./cache'), padding_side='left')

eval_set = datasets.load_dataset('honggen/shp2-test', split='test', data_dir=None, cache_dir='./cache')
all_prompts = [extract_anthropic_prompt(ex['chosen']) for ex in eval_set]
all_chosen = [extract_chosen(ex['chosen']) for ex in eval_set]

all_chosen_list = []
for k in range(0,len(all_chosen),3):
    all_chosen_list.append({
        'instruction': all_prompts[k],
        'output': all_chosen[k],
        'generator': f"harm_chosen"
    })

print('number of prompts', len(set(all_prompts)))
#random.seed(52)
#selected_chosen_list = random.sample(all_chosen_list,30)
random.seed(42)
selected_prompts = random.sample(list(set(all_prompts)),100)

for model_name in model_name_list:
    state_dict = torch.load(f"./cache/honggen/phi_shp-{model_name}/LATEST/policy.pt", map_location='cpu')
    model.module.load_state_dict(state_dict['state'])  # Use model.module to access the wrapped model
    tokenizer.pad_token = tokenizer.eos_token
    model.module.config.pad_token_id = tokenizer.eos_token_id



    all_examples = generator_response(
        model_name,
        model,
        tokenizer,
        selected_prompts,
        max_new_tokens=50,
        temperature=0.85,
        top_k=0,
        top_p=0.50,
        early_stopping=True
    )

    filename = f"./example/phi_shp2/shp_{model_name}100.json"
    with open(filename, 'w') as f:
        json.dump(all_examples, f, indent=4)

    #filename = f"./example/phi_shp2/shp_chosen_sample10.json"
    #with open(filename, 'w') as f:
        #json.dump(selected_chosen_list, f, indent=4)
