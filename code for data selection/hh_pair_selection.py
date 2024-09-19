import transformers
from transformers import BertTokenizer, BertModel
import torch
from torch.nn.functional import cosine_similarity
import datasets
import tqdm
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
from collections import defaultdict
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from utils import get_local_dir
from torch.nn import DataParallel
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data, tokenizer_name = "EleutherAI/pythia-2.8b", max_length=512):
        self.data = data
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name,token='',cache_dir=get_local_dir('./cache'))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt_chosen,prompt_rejected,chosen,rejected = self.data[idx]
        chosen_tokens = self.tokenizer(chosen, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        rejected_tokens = self.tokenizer(rejected, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        return {
            'chosen': prompt_chosen,
            'chosen_tokens': chosen_tokens['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(),
            'rejected': prompt_rejected,
            'rejected_tokens': rejected_tokens['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze()
        }

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]
    
def get_data(split: str,data_dir:str, silent: bool = False, cache_dir: str = None):

    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split='train[:]',data_dir = 'harmless-base', cache_dir=cache_dir)
    print('done')

    data = []
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        #print('-------',row)
        prompt = extract_anthropic_prompt(row['chosen'])
        chosen = row['chosen'][len(prompt):]
        #print('---chosen_response',chosen_response)
        rejected = row['rejected'][len(prompt):]
    
        #print('----prompt',data[prompt])
        data.append((row['chosen'],row['rejected'],chosen,rejected))

    return data

                


device = torch.device('cuda')
model = transformers.AutoModelForCausalLM.from_pretrained('microsoft/phi-1.5',token='',cache_dir=get_local_dir('./cache'), torch_dtype=torch.float16,output_hidden_states = True)

model = model.to(device)
model = torch.nn.DataParallel(model)

data = get_data(split='train' , data_dir = None, cache_dir = './cache/huggingface/datasets')

dataset = TextDataset(data,tokenizer_name='microsoft/phi-1.5')
truncation_mode = 'keep_end'
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
all_chosen = []
all_rejected = []
all_sim = []
for batch in tqdm.tqdm(dataloader):
    #print(batch)
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        
        chosen_outputs = model(batch['chosen_tokens'],attention_mask = batch['chosen_attention_mask'],output_hidden_states=True)
        attention_mask = batch['chosen_attention_mask'].to(device)
        chosen_embeddings = (chosen_outputs.hidden_states[-1] *attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        rejected_outputs = model(batch['rejected_tokens'],attention_mask = batch['rejected_attention_mask'],output_hidden_states=True)
        attention_mask = batch['rejected_attention_mask'].to(device)
        rejected_embeddings = (rejected_outputs.hidden_states[-1] *attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        output = cosine_similarity(chosen_embeddings, rejected_embeddings)
        #print(output)
        all_chosen.extend(batch['chosen'])
        all_rejected.extend(batch['rejected'])
        all_sim.extend(output.cpu().tolist())
        #print(hh)



big_data = {}
big_data['chosen'] = all_chosen
big_data['rejected'] = all_rejected
big_data['sim'] = all_sim
df = pd.DataFrame(big_data)
df_sorted = df.sort_values(by='sim')



num_sample = int(len(df_sorted)/2)

file_name = './data_hh_harm/easy/'+'train_pi.jsonl'
df_sorted[:num_sample].to_json(file_name, orient='records', lines=True)

file_name = './data_hh_harm/hard/'+'train_pi.jsonl'
df_sorted[-num_sample:].to_json(file_name, orient='records', lines=True)

df_sampled = df.sample(frac=0.5)
file_name = './data_hh_harm/random/'+'train_pi.jsonl'
df_sampled.to_json(file_name, orient='records', lines=True)





    
