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

def load_data(data_name, right_case_path, wrong_case_path, args):
    print(right_case_path)
    right_cases = json.load(open(right_case_path))
    right_num = {}
    for item in right_cases:
        right_num[item['num']] = item
    
    wrong_cases = json.load(open(wrong_case_path))
    wrong_num = {}
    for item in wrong_cases:
        wrong_num[item['num']] = item

    data_path = os.path.join(args.data_root,args.dataset, data_name+'.jsonl')
    with open(data_path) as json_file:
        json_list = list(json_file)
    problems = [json.loads(jline) for jline in json_list]
    print(f"number of {data_name} problems: {len(problems)}\n")
    return problems, right_num, wrong_num

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
    parser.add_argument('--train_label_path', type=str, default="")
    parser.add_argument('--extended_train_label_path', type=str, default="")
    parser.add_argument('--dev_label_path', type=str, default="")
    args = parser.parse_args()
    return args
class VerifierTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs['logits']
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
        encoded_ = torch.mean(encoded_, 1)
        return {'logits':self.classifier(encoded_)}

class PintoDataset(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, explanations, tokenizer, source_len, target_len, right_set, wrong_set, args, OnlyAnswer = False, train = False, retain = False
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
        choices_label = ['(a) ','(b) ','(c) ',
                         '(d) ','(e) ']
        def build_train_pair(problem, exp, prompt_format):
            if args.dataset == "strategyqa":
                question = problem['context']
            else:   
                question = problem['question']
            raw_answer = problem['answer']
            answer = problem['choices'][raw_answer]
            choices = problem['choices']
            id =  problem["id"]
            #exp = explanations[id]
            choices_str = ' '.join([label+choice for label, choice in zip(choices_label[:len(choices)],choices)])
            rationale = exp#exp['explanation']
            #rationale = ' '.join([label+choice+'. '+exp for label, choice,exp in zip(choices_label[:len(choices)],choices,problem['explanation'])])

            input_format, output_format = prompt_format.split("-")
            if input_format=="Q":
                input = f"Question: {question} \n Choices: {choices_str}"
            elif input_format=="QE":
                #input = f"Question: {question} \n Choices: {choices}\n Solution: {rationale}"
                input = f"Question: {question} \n Choices: {choices}\n"
            elif input_format=="QA":
                input = f"Question: {question} \n Choices: {choices_str} \n  Answer: The answer is {answer}."
            
            prompt_appendix = ''
            if output_format=="A":
                output = f"Answer: The answer is {answer}."
                prompt_appendix = "\n Answer:"
            elif output_format=="AE":
                output = f"Answer: The answer is {answer}. Because: {rationale}"
                prompt_appendix = "\n Answer:"
            elif output_format=="EA":
                output = f"Solution: {rationale}. Answer: The answer is {answer}. "
                prompt_appendix = "\n Solution:"
            elif output_format=="E":
                output = f"Solution: {rationale}."
                prompt_appendix = "\n Solution:"
            
            prompt = input + prompt_appendix
            pure_answer = f"Answer: The answer is {answer}."
            target = output
            return (prompt, target, pure_answer)
            

        self.tokenizer = tokenizer
        self.problems = problems
        self.source_len = source_len
        self.summ_len = target_len
        if not retain:
            self.target_text = []
            self.source_text = []
            self.label = []
        counter = 0
        right_counter = 0
        wrong_cases_count = len(wrong_set)
        print(len(self.problems))
        for problem in self.problems:
            if counter in wrong_set:
                _label = 0
                exp = wrong_set[counter]['prediction solution']
            else:
                if retain:
                    counter += 1
                    continue
                exp = right_set[counter]['prediction solution']
                _label = 1
                right_counter += 1
                #if train and (right_counter>wrong_cases_count):
                #    continue
            counter += 1
            prompt, target, pure_answer = build_train_pair(problem, exp, 'QE-A')
            self.target_text.append(pure_answer)
            self.source_text.append(prompt)
            self.label.append(_label)
    def extend(
        self, problems, explanations, tokenizer, source_len, target_len, right_set, wrong_set, args, OnlyAnswer = False, train = False, retain = False
    ):
        self.__init__(problems, explanations, tokenizer, source_len, target_len, right_set, wrong_set, args , OnlyAnswer, train, retain = True)

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
        label = torch.tensor([label])
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": label,
        }

        
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
        #encoder.load_state_dict(torch.load(args.pretrained_parameters))
        config = encoder.config
        verifier = Verifier(encoder,config)
        return verifier

    # logging
    console.log(f"[Data]: Reading data...\n")
    # unfold data frame
    train_problems, train_right_cases, train_wrong_cases = load_data("train",os.path.join(args.train_label_path, 'train_right_cases.json'), os.path.join(args.train_label_path, 'train_wrong_cases.json'),args) 
    dev_problems, dev_right_cases, dev_wrong_cases = load_data("test", os.path.join(args.dev_label_path, 'test__right_cases.json'), os.path.join(args.dev_label_path, 'test__wrong_cases.json'), args) 
    
    exp_path = "./ECQA-Dataset/ecqa.jsonl"
    with open(exp_path) as json_file:
        json_list = list(json_file)
    _explanations = [json.loads(jline) for jline in json_list]
    explanations = {}
    for item in _explanations:
        explanations[item['id']] = item
    # Creating the Training and Validation dataset for further creation of Dataloader
    train_set = PintoDataset(
        train_problems,
        explanations,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        train_right_cases,
        train_wrong_cases,
        args,
        train = True
    )
    if args.extended_train_label_path!='':
        args.train_label_path = args.extended_train_label_path
        train_problems, train_right_cases, train_wrong_cases = load_data("train",os.path.join(args.train_label_path, 'train_right_cases.json'), os.path.join(args.train_label_path, 'train_wrong_cases.json'),args)
        train_set.extend(
            train_problems,
            explanations,
            tokenizer,
            model_params["MAX_SOURCE_TEXT_LENGTH"],
            model_params["MAX_TARGET_TEXT_LENGTH"],
            train_right_cases,
            train_wrong_cases,
            args,
            train = True
        )

    eval_set = PintoDataset(
        dev_problems,
        explanations,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        dev_right_cases,
        dev_wrong_cases,
        args,
        OnlyAnswer = True,
    )
    
    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    
    steps = args.steps
    Seq2SeqArgs = Seq2SeqTrainingArguments(
        model_dir,
        evaluation_strategy="steps",
        eval_steps= int(steps/3),
        logging_strategy="epoch",
        save_strategy="steps",
        save_steps= steps,
        learning_rate= model_params["LEARNING_RATE"],
        gradient_accumulation_steps = model_params["gradient_accumulation_steps"],
        per_device_train_batch_size=model_params["TRAIN_BATCH_SIZE"],
        per_device_eval_batch_size=model_params["VALID_BATCH_SIZE"],
        weight_decay=0.01,
        num_train_epochs=model_params["TRAIN_EPOCHS"],
        predict_with_generate=True,
        load_best_model_at_end=True
    )
    datacollator = DataCollatorForSeq2Seq(tokenizer)
    trainer = VerifierTrainer(
        model_init=model_init,
        args=Seq2SeqArgs,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics
    )
    #trainer.evaluate(eval_dataset = eval_set)
    trainer.train()
    trainer.model.save_pretrained(model_dir)

    trainer.evaluate(eval_dataset = eval_set)
    

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
    model_dir = f"./Models/ECQA/{args.dataset}_verifier/{model_name}_{prompt_format}_{timestamp}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_path = f'{model_dir}/log.txt'
    correct_log = open(log_path,'w')
    correct_log.close()
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        print(logits, logits.shape)
        logits = np.argmax(logits,-1)
        print(logits, logits.shape)
        labels = labels.squeeze()
        print(labels, labels.shape)
        correct = sum(logits==labels)/logits.shape[0]
        eval_result = {'correct': correct}
        print(eval_result)
        correct_log = open(log_path,'a')
        correct_log.write(str(eval_result)+'\n')
        correct_log.close()
        return eval_result
    
    T5Trainer(
        args = args,
        model_params=model_params,
        output_dir="outputs",
    )
