import os
import numpy as np
import pandas as pd
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
import re
import json
import argparse
import random
from tqdm import tqdm
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration
from datasets import load_dataset, load_metric
from ScienceQA.models.base_prompt import *
import evaluate
import time
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.functional.text.rouge import rouge_score
random.seed(2023)


path1 = '/home/wuyx/pinto/strategyqa/train.jsonl'
path2 = '/home/wuyx/pinto/strategyqa/strategyqa_train.json'

with open(path1) as json_file:
    json_list = list(json_file)
pinto = [json.loads(jline) for jline in json_list]
origin = json.load(open(path2))
orgin_map = {}
for item in origin:
    orgin_map[item['qid']] = item

res = []
for item in pinto:
    original_item = orgin_map[item['id']]
    item['question']