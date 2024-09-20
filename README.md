# REAL-Alignment
The code for the paper **REAL**: **R**esponse **E**mbedding-based **A**lignment for **L**LMs
![Alt text](./DPO_diagrams.png)

## Data selection
For the given LLMs, we use it to extract the embeddings of response. We have the **(./code for data selection/hh_pair_selection.py)** and **(./code for data selection/shp_pair_selection.py)** to deal with the [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) and [SHP2](https://huggingface.co/datasets/stanfordnlp/SHP-2) dataset.

All the sub-data (hard, easy, centroid) can be found in the [hugging face dataset](https://huggingface.co/datasets/honggen/llama2-help). 

## Supervise Fine-tuning(SFT) and DPO
We follow the setting of the [DPO paper](https://github.com/eric-mitchell/direct-preference-optimization/tree/main). We first supervised the base model using the whole preference answers and randomly sampled answers. Compared to the vanilla DPO code, we use the learning rate scheduler instead of a constant learning rate. 


