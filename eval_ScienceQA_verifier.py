# Importing libraries
from typing import Any, Dict, List, Optional, Tuple, Union

import os
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
# Importing the T5 modules from huggingface/transformers
from torch import nn
from transformers import Trainer
from transformers import BertTokenizer, BertModel
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding, TrainingArguments, Trainer, T5ForConditionalGeneration,PreTrainedModel, PretrainedConfig
from datasets import load_dataset, load_metric
from ScienceQA.models.base_prompt import *
import evaluate
# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console
import time
random.seed(int(time.time()))
from simcse import SimCSE
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
# define a rich console logger
console = Console(record=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='ScienceQA/data/scienceqa')
    parser.add_argument('--output_root', type=str, default='./results')
    parser.add_argument('--caption_file', type=str, default='ScienceQA/data/captions.json')
    parser.add_argument('--model', type=str, default="allenai/unifiedqa-t5-base")

    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    # user options
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--train', type=str, default='True', choices=['True','False'])
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])
    parser.add_argument('--test_number', type=int, default=10, help='GPT-3 is expensive. -1 for whole val/test set')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    #parser.add_argument('--compute_metrics', action='store_true')
    #parser.add_argument('--save_every', type=int, default=10, help='Save the result with every n examples.')
    #parser.add_argument('--debug', action='store_true')
    parser.add_argument('--train_strategy',
                        type=str,
                        default='random_sample',
                        choices=[
                            'random_sample', 'most_similar', 'replace', 'generated'
                            ],
                        help='train strategy')
    # format 
    # generated_data_dir/
    # ---- train_wrong_cases.json
    # ---- test_wrong_cases.json
    # ---- eval_wrong_cases.json
    parser.add_argument('--generated_data_dir', type=str, default='')
    parser.add_argument('--prompt_format',
                        type=str,
                        default='CQME',
                        choices=[
                            'CQME','QCMLE','QCMEL'
                        ],
                        help='prompt format template')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--valid_epochs', type=int, default=1)
    parser.add_argument('--test_epochs', type=int, default=1)
    parser.add_argument('--train_batchsize', type=int, default=4)
    parser.add_argument('--valid_batchsize', type=int, default=32)
    parser.add_argument('--test_batchsize', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--weight_decay',type=float, default=0.01)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--prepared_match_dir', type=str, default=None)
    parser.add_argument('--verifier_path', type=str, default='')
    parser.add_argument('--model_verifier', type=str, default="allenai/unifiedqa-t5-base")
    parser.add_argument('--cases_dir', type=str, default='')
    parser.add_argument('--add_wrong', action="store_true")
    parser.add_argument('--add_right', action="store_true")
    
    #parser.add_argument('--temperature', type=float, default=0.0)
    #parser.add_argument('--top_p', type=float, default=1.0)
    #parser.add_argument('--frequency_penalty', type=float, default=0.0)
    #parser.add_argument('--presence_penalty', type=float, default=0.0)

    args = parser.parse_args()
    return args 

if __name__ == '__main__': 
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)
    timestamp = time.strftime("%m-%d-%H", time.localtime())
    train_strategy = args.train_strategy
    if train_strategy=='generated':
        train_strategy += "_" + args.generated_data_dir.split('/')[-1]
    model_dir = '/'.join(args.verifier_path.split('/')[:-1]) 
    logging_dir = f"{model_dir}/log"
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
    # Setting up the device for GPU usage
    from torch import cuda
    device = f'cuda:{args.device}' if cuda.is_available() else 'cpu'
    cases_dir = '-'.join(args.cases_dir.split('/')[-2:])
    log_path = f'{model_dir}/correct_log_{timestamp}_{cases_dir}_{args.add_wrong}_{args.add_right}.txt'
    correct_log = open(log_path,'w')
    correct_log.write("args "+str(args)+'\n')
    correct_log.write('====Input Arguments====\n')
    correct_log.write(json.dumps(vars(args), indent=2, sort_keys=False)+'\n')
    correct_log.close()
    
    if args.train_strategy=="most_similar" and args.prepared_match_dir==None:
        # Import our models. The package will take care of downloading the models automatically
        sim_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
        sim_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)

class VerifierTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = torch.mean(outputs['logits'],1)
        loss_fct = nn.CrossEntropyLoss()
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
        return {'logits':self.classifier(encoded_)}

def load_data(args):
    problems = json.load(open(os.path.join(args.data_root, 'problems.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json')))
    captions = json.load(open(args.caption_file))["captions"]

    for qid in problems:
        problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")
    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids

class ScienceQADataset(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, tokenizer, source_len, target_len, split, args, OnlyAnswer = False
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.label = []
        self.source_text = []
        cnt = 0 
        if args.cases_dir=='':
            for qid in self.data:
                #print(cnt)
                question, context, choice, answer, lecture, solution = build_data(problems, qid, args)
                prompt = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
                self.label.append(1)
                self.source_text.append(prompt)
                cnt+=1
        else:
            if args.add_right:
                right_cases = json.load(open(f"{args.cases_dir}/{split[:-4]}_right_cases.json"))
                for case in right_cases:
                    qid = qids[case['num']]
                    question, context, choice, answer, lecture, solution = build_data(problems, qid, args)
                    _solution = case["prediction solution"].replace('Solution: ','')
                    _prompt = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {_solution}\n"
                    self.label.append(1)
                    self.source_text.append(_prompt)
            if args.add_wrong:
                wrong_cases = json.load(open(f"{args.cases_dir}/{split[:-4]}_wrong_cases.json"))
                for case in wrong_cases:
                    qid = qids[case['num']]
                    question, context, choice, answer, lecture, solution = build_data(problems, qid, args)
                    _solution = case["prediction solution"].replace('Solution: ','')
                    _prompt = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {_solution}\n"
                    self.label.append(0)
                    self.source_text.append(_prompt)
            
    def __len__(self):
        """returns the length of dataframe"""

        return len(self.source_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        label = self.label[index]

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        label = torch.tensor(label)

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": label,
        }

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    logits = np.mean(logits,1)
    logits = np.argmax(logits,-1)
    correct = sum(logits==labels)/logits.shape[0]
    eval_result = {'correct': correct}
    print(eval_result)
    correct_log = open(log_path,'a')
    correct_log.write(str(eval_result)+'\n')
    correct_log.close()
    return eval_result

        
def T5Trainer(
    dataframe, args, model_params
):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenizer for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    def model_init():
        verifier_encoder = AutoModelForSeq2SeqLM.from_pretrained(args.model_verifier)
        config = verifier_encoder.config
        verifier = Verifier(verifier_encoder,config)
        verifier.load_state_dict(torch.load(args.verifier_path))
        return verifier

    # logging
    console.log(f"[Data]: Reading data...\n")
    # unfold data frame
    problems = dataframe['problems']
    qids = dataframe['qids']
    train_qids = qids['train']
    test_qids = qids['test']
    val_qids = qids['val']
    print("================== load train ==================")
    # Creating the Training and Validation dataset for further creation of Dataloader
    '''
    train_set = ScienceQADataset(
        problems,
        train_qids,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        'train_set',
        args
    )
    '''
    print("================== load eval ==================")
    eval_set = ScienceQADataset(
        problems,
        val_qids,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        'eval_set',
        args,
    )
    print("================== load test ==================")
    test_set = ScienceQADataset(
        problems,
        test_qids,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        'test_set',
        args,
    )
    print("================== load finish ==================")
    steps = 4074
    Seq2SeqArgs = TrainingArguments(
        model_dir,
        evaluation_strategy="steps",
        eval_steps= int(steps/2),
        logging_strategy="epoch",
        logging_dir = logging_dir,
        save_strategy="steps",
        save_steps= steps,
        learning_rate= model_params["LEARNING_RATE"],
        gradient_accumulation_steps = model_params["gradient_accumulation_steps"],
        per_device_train_batch_size=model_params["TRAIN_BATCH_SIZE"],
        per_device_eval_batch_size=model_params["VALID_BATCH_SIZE"],
        weight_decay=args.weight_decay,
        num_train_epochs=model_params["TRAIN_EPOCHS"],
        load_best_model_at_end=True
    )
    datacollator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    trainer = VerifierTrainer(
        model_init=model_init,
        args=Seq2SeqArgs,
        train_dataset=eval_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics
    )
    trainer.evaluate(eval_dataset = test_set)
    

if __name__ == '__main__':
    problems, qids = load_data(args)  # probelms, test question ids, shot example ids
    dataframe = {'problems':problems, 'qids':qids}


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

    T5Trainer(
        dataframe=dataframe,
        args = args,
        model_params=model_params,
    )
