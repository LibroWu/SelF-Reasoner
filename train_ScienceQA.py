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
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration
from datasets import load_dataset, load_metric
from ScienceQA.models.base_prompt import *
import evaluate
# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console
import time
random.seed(int(time.time()))

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
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain','train_no_image'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival','val_no_image','test_no_image'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest','test_no_image'])
    parser.add_argument('--test_number', type=int, default=10, help='GPT-3 is expensive. -1 for whole val/test set')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--compute_metrics', action='store_true')
    #parser.add_argument('--save_every', type=int, default=10, help='Save the result with every n examples.')
    #parser.add_argument('--debug', action='store_true')
    parser.add_argument('--prompt_format',
                        type=str,
                        default='CQM-A',
                        choices=[
                            'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE', 'QCM-A',
                            'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE', 'QCML-A', 'QCME-A',
                            'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE','QCM-E','QCMA-E', 'QCM[E]-A','QCM[LE]-A'
                        ],
                        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=3, help='Number of n-shot training examples.')
    parser.add_argument('--shot_qids', type=list, default=None, help='Question indexes of shot examples')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--valid_epochs', type=int, default=1)
    parser.add_argument('--test_epochs', type=int, default=1)
    parser.add_argument('--save_steps', type=int, default=21210)
    parser.add_argument('--eval_steps', type=int, default=int(21210/2)) #assert save_steps % eval_steps == 0
    parser.add_argument('--train_batchsize', type=int, default=4)
    parser.add_argument('--valid_batchsize', type=int, default=32)
    parser.add_argument('--test_batchsize', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--weight_decay',type=float, default=0.01)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--OnlyAnswer',action='store_true')
    #parser.add_argument('--temperature', type=float, default=0.0)
    #parser.add_argument('--top_p', type=float, default=1.0)
    #parser.add_argument('--frequency_penalty', type=float, default=0.0)
    #parser.add_argument('--presence_penalty', type=float, default=0.0)

    args = parser.parse_args()
    assert (args.save_steps % args.eval_steps == 0)
    return args 

if __name__ == '__main__': 
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)
    timestamp = time.strftime("%m-%d-%H", time.localtime())
    model_dir = f"./Models/ScienceQA/{args.model}_{args.prompt_format}_{args.train_split}_{timestamp}"
    logging_dir = f"./Models/ScienceQA/{args.model}_{args.prompt_format}_{timestamp}/logs"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
    # Setting up the device for GPU usage
    from torch import cuda
    device = f'cuda:{args.device}' if cuda.is_available() else 'cpu'
    log_path = f'{model_dir}/correct_log.txt'
    correct_log = open(log_path,'w')
    correct_log.write("args "+str(args)+'\n')
    correct_log.write('====Input Arguments====\n')
    correct_log.write(json.dumps(vars(args), indent=2, sort_keys=False)+'\n')
    correct_log.close()
    



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

    # pick up shot examples from the training set
    shot_qids = args.shot_qids
    if shot_qids == None:
        assert args.shot_number >= 0 and args.shot_number <= 32
        shot_qids = random.sample(train_qids, args.shot_number)  # random sample
    else:
        shot_qids = [str(qid) for qid in shot_qids]
        for qid in shot_qids:
            assert qid in train_qids  # check shot_qids
    print("training question ids for prompting: ", shot_qids, "\n")
    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids, shot_qids

class ScienceQADataset(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, tokenizer, source_len, target_len, args, OnlyAnswer = False
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
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        for qid in self.data:
            problem = problems[qid]
            if args.prompt_format=='QCM[E]-A' or args.prompt_format=='QCM[LE]-A':
                QCM_prompt, QCME_prompt, target, pure_answer = build_train_pair(problems, [], qid, args)
                self.target_text.append(target)
                self.source_text.append(QCM_prompt)
                self.target_text.append(target)
                self.source_text.append(QCME_prompt)
            else:
                prompt, target, pure_answer = build_train_pair(problems, [], qid, args)
                if OnlyAnswer:
                    self.target_text.append(pure_answer)
                    self.source_text.append(prompt+"Let's think step by step!")
                else:
                    self.target_text.append(target)
                    self.source_text.append(prompt+"Let's think step by step!")
            
                

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()
        #target_mask = target["attention_mask"].squeeze()

        #self.set_format(type="torch", columns=[ "input_ids", "attention_mask", "labels"])
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
        }

def compute_metrics(eval_preds):
    def compute(predictions,references):
        correct = 0
        for pred,ref in zip(predictions, references):
            l = len(ref)
            i = l-1
            while (i>0 and (ref[i]==0 or ref[i]==5 or ref[i]==1)):
                i=i-1
            match_target = ref[2:i+1]
            l_target = len(match_target)
            i = len(pred) - l_target
            matched = False
            while i>0:
                if ((match_target==pred[i:i+l_target]).all()):
                    matched = True
                    break
                i = i - 1
            if matched:
                correct = correct+ 1
        
        return {'correct':correct/len(references)}
    logits, labels = eval_preds
    eval_result = compute(predictions=logits, references=labels)
    correct_log = open(log_path,'a')
    correct_log.write(str(eval_result)+'\n')
    correct_log.close()
    print(eval_result)
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

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    def model_init():
        return AutoModelForSeq2SeqLM.from_pretrained(model_params["MODEL"]) 
        
    # logging
    console.log(f"[Data]: Reading data...\n")
    # unfold data frame
    problems = dataframe['problems']
    qids = dataframe['qids']
    train_qids = qids['train']
    test_qids = qids['test']
    val_qids = qids['val']
    
    # Creating the Training and Validation dataset for further creation of Dataloader
    train_set = ScienceQADataset(
        problems,
        train_qids,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        args,
    )
    eval_set = ScienceQADataset(
        problems,
        val_qids,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        args,
        OnlyAnswer = args.OnlyAnswer,
    )
    
    test_set = ScienceQADataset(
        problems,
        test_qids,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        args,
        OnlyAnswer = args.OnlyAnswer,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    test_params = {
        "batch_size": model_params["TEST_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }
    
    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    #training_loader = DataLoader(training_set, **train_params)
    #val_loader = DataLoader(val_set, **val_params)
    #test_loader = DataLoader(test_set, **test_params)
    model_name = model_params["MODEL"]
    prompt_format = args.prompt_format
    lr = str(model_params["LEARNING_RATE"])
    
    

    save_steps = model_params['SAVE_STEPS']
    eval_steps = model_params['EVAL_STEPS']
    Seq2SeqArgs = Seq2SeqTrainingArguments(
        model_dir,
        evaluation_strategy="steps",
        eval_steps= eval_steps,
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
        predict_with_generate=True,
        load_best_model_at_end=True
    )
    datacollator = DataCollatorForSeq2Seq(tokenizer)
    trainer = CoTSeq2SeqTrainer(
        model_init=model_init,
        args=Seq2SeqArgs,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics if args.compute_metrics else None
    )

    trainer.train()
    trainer.model.save_pretrained(model_dir)

    trainer.evaluate(eval_dataset = test_set)
    

if __name__ == '__main__':
    problems, qids, shot_qids = load_data(args)  # probelms, test question ids, shot example ids
    dataframe = {'problems':problems, 'qids':qids, 'shot_qids':shot_qids}


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
        "SAVE_STEPS": args.save_steps, # steps for saving
        "EVAL_STEPS": args.eval_steps, # steps for evaluation
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
