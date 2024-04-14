# Importing libraries
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
import re
import json
import argparse
import random
from tqdm import tqdm
from torch import nn
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding, TrainingArguments, Trainer, T5ForConditionalGeneration,PreTrainedModel, PretrainedConfig,  DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration
from datasets import load_dataset, load_metric
import evaluate
# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

# define a rich console logger
console = Console(record=True)

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

class CoTSeq2SeqTrainer(Seq2SeqTrainer):
     def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`List[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is ``"eval"`` (default)
            max_length (:obj:`int`, `optional`):
                The maximum target length to use when predicting with the generate method.
            num_beams (:obj:`int`, `optional`):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        max_length = 512 if max_length is None else max_length 
        num_beams = 2 if num_beams is None else num_beams
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix,max_length = max_length,num_beams = num_beams)


def load_data(data_name,args):
    data_path = os.path.join(args.data_root,args.dataset, data_name+'.jsonl')
    with open(data_path) as json_file:
        json_list = list(json_file)
    problems = [json.loads(jline) for jline in json_list]
    print(f"number of {data_name} problems: {len(problems)}\n")
    return problems

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./pinto/')
    parser.add_argument('--train', type=str, default='True', choices=['True','False'])
    parser.add_argument('--train_split', type=str, default='train', choices=['train'])
    parser.add_argument('--val_split', type=str, default='dev', choices=['dev'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test'])
    parser.add_argument('--dataset', type=str, default='csqa', choices=['csqa', 'strategyqa', 'qasc', 'PaxHeader', 'obqa'])
    parser.add_argument('--prompt_format',
                        type=str,
                        default='Q-A',
                        choices=[
                            'Q-A','QE-A','Q-AE','Q-EA','Q-E','Q[E]-A'
                        ],
                        help='prompt format template')
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--steps', type=int, default=12450)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--valid_epochs', type=int, default=1)
    parser.add_argument('--test_epochs', type=int, default=1)
    parser.add_argument('--train_batchsize', type=int, default=4)
    parser.add_argument('--valid_batchsize', type=int, default=32)
    parser.add_argument('--test_batchsize', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--weight_decay',type=float, default=0.01)
    parser.add_argument('--compute_metrics', action='store_true')
    parser.add_argument('--model', type=str, default="allenai/unifiedqa-t5-base")
    parser.add_argument('--pretrained_parameters', type=str, default='')
    parser.add_argument('--path_QE_A_right', type=str, default='path_to_the_right_cases_using_Rationale.json') # right cases using rationale; Q->QE->A
    parser.add_argument('--path_QE_A_wrong', type=str, default='path_to_the_wrong_cases_using_Rationale.json') # wrong cases using rationale; Q->QE->A
    parser.add_argument('--path_Q_A_right', type=str, default='path_to_the_right_cases_using_Question_Only.json') # right cases using Question Only; Q->A
    
    args = parser.parse_args()
    return args
class VerifierTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = torch.mean(outputs['logits'],1)
        #print("logits", logits, logits.shape, logits.view(-1, 2).shape)
        #print("labels", labels, labels.shape,labels.view(-1).shape)
        loss_fct = nn.CrossEntropyLoss()
        #loss_fct = nn.CrossEntropyLoss(weight = torch.tensor([4.598,0.561]).to(device))
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class Verifier(PreTrainedModel):
    def __init__(self, encoder, config):
        super(Verifier, self).__init__(config)
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.hidden_size,2)

    def forward(self, 
            input_ids=None,
            attention_mask=None,
            labels=None,
            ):
        encoded_ = self.encoder.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        #return {'logits':self.classifier(torch.mean(encoded_,1))}
        return {'logits':torch.mean(self.classifier(encoded_), 1)}
        
def T5Trainer(
    args, model_params, output_dir="./outputs/"
):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    def model_init():
        encoder = AutoModelForSeq2SeqLM.from_pretrained(model_params["MODEL"]) 
        config = encoder.config
        verifier = Verifier(encoder,config)
        verifier.load_state_dict(torch.load(args.pretrained_parameters))
        return verifier
    verifier = model_init()
    path_QE_A_right = args.path_QE_A_right 
    path_QE_A_wrong = args.path_QE_A_wrong 
    path_Q_A_right = args.path_Q_A_right
    
    with open(path_QE_A_wrong, 'r') as fcc_file:
        cases_QE_A_wrong = json.load(fcc_file)
    wrong_set = {}
    for case in cases_QE_A_wrong:
        wrong_set[case['num']] = case

    dev_problems = load_data("test",args) 
    with open(path_QE_A_right, 'r') as fcc_file:
        cases_QE_A_right = json.load(fcc_file)
    num_set_QE_A_right = []
    for case in cases_QE_A_right:
        num_set_QE_A_right.append(case['num'])
    num_set_QE_A_right = set(num_set_QE_A_right)
    with open(path_Q_A_right, 'r') as fcc_file:
        cases_Q_A_right = json.load(fcc_file)
    num_set_Q_A_right = []
    for case in cases_Q_A_right:
        num_set_Q_A_right.append(case['num'])
    num_set_Q_A_right = set(num_set_Q_A_right)

    hardest_num_set = num_set_QE_A_right.intersection(num_set_Q_A_right)
    len_overlap = len(hardest_num_set)
    #len_overlap = len(num_set_QE_A_right.intersection(num_set4))
    print(path_QE_A_right+':',len(num_set_QE_A_right))
    print(path_Q_A_right+':',len(num_set_Q_A_right))
    print("overlap:",len_overlap)
    only_Q_A = []
    only_QE_A = []
    for case in cases_QE_A_right:
        if case['num'] not in num_set_Q_A_right:
            only_QE_A.append(case)
    for case in cases_Q_A_right:
        if case['num'] not in num_set_QE_A_right:
            only_Q_A.append(case)
    choices_label = ['(a) ','(b) ','(c) ',
                         '(d) ','(e) ']
    metric = 0
    print('only_Q_A', len(only_Q_A))
    print('only_QE_A', len(only_QE_A))
    _only_QE_A = []
    _only_Q_A = []
    for case in tqdm(only_Q_A):
        problem = dev_problems[case['num']]
        choices = problem['choices']
        choices_str = ' '.join([label+choice for label, choice in zip(choices_label[:len(choices)],choices)])
        question = case['question']
        case['prediction solution'] = wrong_set[case['num']]['prediction solution']
        rationale = case['prediction solution']
        input = f"Question: {question} \n Choices: {choices}\n Solution: {rationale}"
        source = tokenizer.batch_encode_plus(
            [input],
            max_length=512,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"]
        source_mask = source["attention_mask"]
        logits = verifier.forward(
            input_ids=source_ids,
            attention_mask=source_mask,
            )['logits']
        logits = torch.argmax(logits,-1)
        if logits[0]==0:
            metric += 1
        case['verifier'] = int(logits[0])
        _only_Q_A.append(case)
    print(metric)
    for case in tqdm(only_QE_A):
        problem = dev_problems[case['num']]
        choices = problem['choices']
        choices_str = ' '.join([label+choice for label, choice in zip(choices_label[:len(choices)],choices)])
        question = case['question']
        rationale = case['prediction solution']
        input = f"Question: {question} \n Choices: {choices}\n Solution: {rationale}"
        source = tokenizer.batch_encode_plus(
            [input],
            max_length=512,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"]
        source_mask = source["attention_mask"]
        logits = verifier.forward(
            input_ids=source_ids,
            attention_mask=source_mask,
            )['logits'] 
        logits = torch.argmax(logits,-1)
        if logits[0]==0:
            metric -= 1
        case['verifier'] = int(logits[0])
        _only_QE_A.append(case)
    print(metric)
    with open(f"ecqa_test_only_Q-A.json","w") as os:
            json.dump(_only_Q_A,os)
    with open(f"ecqa_test_only_QE-A.json","w") as os:
            json.dump(_only_QE_A,os)


if __name__ == '__main__':
    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )
    
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    

    # let's define model parameters specific to T5
    model_params = {
        "MODEL": args.model,  # model_type: t5-base/t5-large
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "TRAIN_BATCH_SIZE": args.train_batchsize,  # training batch size
        "VALID_BATCH_SIZE": args.valid_batchsize,  # validation batch size
        "TEST_BATCH_SIZE": args.test_batchsize,  # validation batch size
        "TRAIN_EPOCHS": args.train_epochs,  # number of training epochs
        "VAL_EPOCHS": args.valid_epochs,  # number of validation epochs
        "TEST_EPOCHS": args.test_epochs,  # number of validation epochs
        "LEARNING_RATE": args.lr,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": args.max_tokens,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": args.max_tokens,  # max length of target text
        "SEED": args.seed,  # set seed for reproducibility
    }

    model_name = model_params["MODEL"]
    prompt_format = args.prompt_format
    lr = str(model_params["LEARNING_RATE"])
    timestamp = time.strftime("%m-%d-%H", time.localtime())
    
    T5Trainer(
        args = args,
        model_params=model_params,
        output_dir="outputs",
    )
