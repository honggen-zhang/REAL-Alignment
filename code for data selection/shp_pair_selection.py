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
from utils import (
    get_local_dir,TemporarilySeededRandom
)
from torch.nn import DataParallel
#from preference_datasets import tokenize_batch_element
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.spatial.distance import cdist
def cluster_center(embeddings):
    max_sim = -1
    max_sim_ind = []
    min_sim = 1
    min_sim_ind = []
    print(len(embeddings))
    for i in range(len(embeddings)-1):
        for j in range(i+1,len(embeddings)):
            #print(torch.tensor(embeddings[i]))
            #print(torch.tensor(embeddings[j]))
            similarity = cosine_similarity(torch.tensor(embeddings[i]).unsqueeze(0), torch.tensor(embeddings[j]).unsqueeze(0))
            #print(similarity)
            if similarity>max_sim:
                max_sim = similarity
                max_sim_ind = [i,j]
            if similarity<min_sim:
                min_sim = similarity
                min_sim_ind = [i,j]
    #print('max--',max_sim)
    #print('min--',min_sim)
    #threshold = max_sim[0]-min_sim[0]
    #if threshold.item()<0.1:
        #print('done')
        #return None
        
            
    
    #print(len(embeddings))
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(embeddings)


    centroids = kmeans.cluster_centers_
    #print('centroids',centroids)

     # Calculate the distance between each sample and each centroid
    distances = cdist(embeddings, centroids, 'euclidean')
    #print('euclidean',distances)

    # Find the index of the minimum distance to each centroid
    closest_indices = distances.argmin(axis=0)
    #print(closest_indices)
    centroid_indx = [min(closest_indices),max(closest_indices)]
    #print('centroid--',centroid_indx)
    if len(embeddings) > 3:
        random_indx = [1,2]
    if len(embeddings) == 3:
        random_indx = random.choice([[0,1],[0,2],[1,2]])
    if len(embeddings) <= 2:
        random_indx = [0,1]
    return max_sim_ind,min_sim_ind,centroid_indx,random_indx

    

def tokenize_batch_element(prompt: str, chosen: str, rejected: str, chosen_s: float, rejected_s: float, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    #print(chosen_tokens)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    #prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    #assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)
    

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))


    # if that's still too long, truncate the response
    if longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: rejected_tokens[k] for k in rejected_tokens}
    #chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    #chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    #rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    #rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen_scores'] = chosen_s
    batch['rejected_scores'] = rejected_s
    batch['chosen'] = prompt+chosen
    batch['rejected'] = prompt+rejected
    #batch['chosen_response_only'] = chosen
    #batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens
    #print(batch)

    return batch




def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:

    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            #print(k)
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                #print(batch[0])
                if 'prompt' in k:  
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                #print(padded_batch[k])
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn
def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]
    
def get_hh(split: str,data_dir:str, silent: bool = False, cache_dir: str = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str],List[float], str]]]:

    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('SHP-2-all', split='train', data_dir=data_dir, cache_dir=cache_dir)
    print('done')

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        #print('---prompt',prompt)
        chosen_response = ex['chosen'][len(prompt):]
        #print('---chosen_response',chosen_response)
        rejected_response = ex['rejected'][len(prompt):]
        #print('---rejected_response',rejected_response)
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        #print('-------',row)
        prompt, chosen, rejected = split_prompt_and_responses(row)

        scores = [row['chosen_score'], row['rejected_score']]
        
        #sim_score = calculate_similarity(chosen, rejected, tokenizer, model)
        responses = [chosen, rejected]
        
        n_responses = len(data[prompt]['responses'])
        #print('----prompt',data[prompt])
        if len(data[prompt]['pairs'])>=2:
            continue
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['scores'].extend(scores)
        #data[prompt]['sft_target'] = chosen
        #print(data[prompt])
        
        #print(prompt)

    return data



def data_generator(tokenizer,flat_data):
    
    #permutation_seeds = np.random.randint(0, 2**16)
    #permutation_seeds = 42
    #with TemporarilySeededRandom(permutation_seeds):
        #random.shuffle(flat_data)
    #tokenizer = transformers.AutoTokenizer.from_pretrained('EleutherAI/pythia-2.8b', cache_dir=get_local_dir('./cache'))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    collate_fn = get_collate_fn(tokenizer)

    example_idx = 0
    done = False
    batch = []
    for prompt, responses, pairs, scores, truncation_mode in flat_data:
        for p in pairs:
            #print(prompt[:20])
            if done:
                break
            batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], scores[p[0]],scores[p[1]], truncation_mode, tokenizer, max_length=128, max_prompt_length=128)
            #print(batch_element)
            batch.append(batch_element)
            example_idx += 1
            if len(batch) == batch_size:
                #print(batch)
                #print(example_idx)
                yield collate_fn(batch)
                if example_idx >= n_examples:
                    done = True
                batch = []
                
device = torch.device('cuda')
torch.cuda.empty_cache()
#model_name = "bert-base-uncased"
batch_size = 8
n_examples = 140000
print('loading model')
#model = transformers.AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token='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',cache_dir=get_local_dir('./cache'),output_hidden_states = True)
    
#tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token='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',cache_dir=get_local_dir('./cache'))

model = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-2.8b",cache_dir=get_local_dir('./cache'),output_hidden_states = True)
    
tokenizer = transformers.AutoTokenizer.from_pretrained( "EleutherAI/pythia-2.8b",cache_dir=get_local_dir('./cache'))


#model_name = "bert-large-uncased"
#model = BertModel.from_pretrained(model_name)
#tokenizer = BertTokenizer.from_pretrained(model_name)

#state_dict = torch.load('./cache/honggen/pythia_helpbase-sft/LATEST/policy.pt', map_location='cpu')
#model.load_state_dict(state_dict['state'])

model = torch.nn.DataParallel(model).cuda()
model.eval()
def load_data(split,model):
    get_dataset = get_hh(split=split , data_dir = None, cache_dir = './cache/huggingface/datasets')
    truncation_mode = 'keep_end'
    flat_data = []
    for prompt, data in get_dataset.items():
        flat_data.append((prompt, data['responses'], data['pairs'], data['scores'], truncation_mode))
    
    sim_collect_list = []
    big_data = {}
    all_chosen = []
    all_rejected = []
    all_sim = []
    data_emb = defaultdict(dict)
    data_score = defaultdict(dict)
    #print(flat_data[:5])
    for batch in data_generator(tokenizer,flat_data):
        #print('-----',batch)
        with torch.no_grad():
            chosen_outputs = model(batch['chosen_input_ids'],attention_mask=batch['chosen_attention_mask'],output_hidden_states=True)
            attention_mask = batch['chosen_attention_mask'].to(device)
            #print(chosen_outputs.hidden_states[-1].device)
            #print(attention_mask)
            #print(chosen_outputs.hidden_states[-1].shape)
            chosen_embeddings = (chosen_outputs.hidden_states[-1] *attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    
        # Use the mean of the last layer's features
            #chosen_embeddings = chosen_outputs.hidden_states[-1].mean(dim=1)
            
            rejected_outputs = model(batch['rejected_input_ids'],attention_mask=batch['rejected_attention_mask'],output_hidden_states=True)
            attention_mask = batch['rejected_attention_mask'].to(device)
            #print(attention_mask.shape)
            #print(rejected_outputs.hidden_states[-1].shape)
            rejected_embeddings = (rejected_outputs.hidden_states[-1] *attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    
            #rejected_embeddings = rejected_outputs.hidden_states[-1].mean(dim=1)
            for i in range(batch_size):
                chosen_embeddings_norm = torch.norm(chosen_embeddings[i], p=2).cpu()
                rejected_embeddings_norm = torch.norm(rejected_embeddings[i], p=2).cpu()
                #print(rejected_embeddings_norm)
                data_emb[batch['prompt'][i]][batch['chosen'][i]] = chosen_embeddings[i].cpu()/chosen_embeddings_norm
                data_emb[batch['prompt'][i]][batch['rejected'][i]]= rejected_embeddings[i].cpu()/rejected_embeddings_norm

                data_score[batch['prompt'][i]][batch['chosen'][i]] = batch['chosen_scores'][i]
                data_score[batch['prompt'][i]][batch['rejected'][i]]= batch['rejected_scores'][i]
                #print(batch['chosen'][i],chosen_embeddings[i].cpu()/chosen_embeddings_norm)
                #print(batch['rejected'][i],rejected_embeddings[i].cpu()/rejected_embeddings_norm)
                #print('============')
            
            #similarity = cosine_similarity(chosen_embeddings, rejected_embeddings)
            #print(similarity.cpu().numpy())
            #all_sim.extend(list(similarity.cpu().numpy()))
    #responses = []
            #print(data_emb[batch['prompt'][i]][batch['chosen'][i]])


    return data_emb,data_score

    



#train_list = ['train[:20001]','train[20001:40008]','train[40008:60000]','train[60000:80007]','train[80007:100002]','train[100002:120000]','train[120000:]']

train_list = ['train[:]']
for i in range(1):
    all_chosen_hard = []
    all_rejected_hard = []
    all_chosen_easy = []
    all_rejected_easy = []
    all_chosen_centroid = []
    all_rejected_centroid = []
    all_chosen_random = []
    all_rejected_random = []
    data_emb,data_score = load_data(train_list[i],model)
    for prompt, dict_emd in data_emb.items():
        #print('========',dict_emd)
        embds = []
        responses = []
        scores = []
        for resp,emb in dict_emd.items():
            #print('+++++')
            responses.append(resp)
            embds.append(emb.cpu().numpy())
            scores.append(data_score[prompt][resp])
        #print(scores)
        #print(responses)
        #print(embds)
        if cluster_center(embds) is not None:
            hard_id, easy_id, centroid_id, random_id = cluster_center(embds)
            #print(hard_id,centroid_id)
            all_chosen_hard.append(responses[hard_id[0]])
            all_rejected_hard.append(responses[hard_id[1]])
    
            all_chosen_easy.append(responses[easy_id[0]])
            all_rejected_easy.append(responses[easy_id[1]])
    
            all_chosen_centroid.append(responses[centroid_id[0]])
            all_rejected_centroid.append(responses[centroid_id[1]])
    
            all_chosen_random.append(responses[random_id[0]])
            all_rejected_random.append(responses[random_id[1]])
        #similarity = cosine_similarity(embds[chosen_id], embds[rejected_id])
    del dict_emd
    torch.cuda.empty_cache()
    big_data = {}
    big_data['chosen'] = all_chosen_centroid
    big_data['rejected'] = all_rejected_centroid
    df = pd.DataFrame(big_data)
    file_name = 'data_hh_harm/centroid/'+'train_pa'+str(i)+'.jsonl'
    df.to_json(file_name, orient='records', lines=True)
    print(len(df))

    big_data = {}
    big_data['chosen'] = all_chosen_hard
    big_data['rejected'] = all_rejected_hard
    df = pd.DataFrame(big_data)
    file_name = './data_hh_harm/hard/'+'train_pa'+str(i)+'.jsonl'
    df.to_json(file_name, orient='records', lines=True)

    big_data = {}
    big_data['chosen'] = all_chosen_easy
    big_data['rejected'] = all_rejected_easy
    df = pd.DataFrame(big_data)
    file_name = './data_hh_harm/easy/'+'train_pa'+str(i)+'.jsonl'
    df.to_json(file_name, orient='records', lines=True)

    big_data = {}
    big_data['chosen'] = all_chosen_random
    big_data['rejected'] = all_rejected_random
    df = pd.DataFrame(big_data)
    file_name = './data_hh_harm/random/'+'train_pa'+str(i)+'.jsonl'
    df.to_json(file_name, orient='records', lines=True)
#response_emb_data['response'] = responses
#response_emb_data['embs'] = embds
        

#df = pd.DataFrame(response_emb_data)
#file_name = './models_sim/checkpoint_emb_0.jsonl'
#df.to_json(file_name, orient='records', lines=True)
#df.to_csv(file_name)

