# Importing libraries
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
import re
import json
import argparse
from tqdm import tqdm
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration
from datasets import load_dataset, load_metric
from ScienceQA.models.base_prompt import *
import evaluate

import random
import time
random.seed(int(time.time()))

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

# define a rich console logger
console = Console(record=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./LastLetter/')
    parser.add_argument('--train', type=str, default='True', choices=['True','False'])
    parser.add_argument('--train_split', type=str, default='train', choices=['train'])
    parser.add_argument('--val_split', type=str, default='dev', choices=['dev'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test'])
    
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
    parser.add_argument('--solution_giver_path', type=str, default='')
    parser.add_argument('--answer_extractor_path', type=str, default="")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--use_pipeline', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    assert args.do_eval or args.do_test or args.do_train
    random.seed(args.seed)
    target = ''
    if args.do_eval:
        target += 'eval'
    if args.do_test:
        target += 'test'
    if args.do_train:
        target += 'train'
    timestamp = time.strftime("%m-%d-%H", time.localtime())
    
    log_path = args.answer_extractor_path.replace('pytorch_model.bin',f'correct_log_pipeline_{target}_{timestamp}.txt')
    correct_log = open(log_path,'w')
    correct_log.write("args "+str(args)+'\n')
    correct_log.write('====Input Arguments====\n')
    correct_log.write(json.dumps(vars(args), indent=2, sort_keys=False)+'\n')
    correct_log.close()
    
    # Setting up the device for GPU usage
    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)

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
    data_path = os.path.join(args.data_root, data_name+'.json')
    problems = json.load(open(data_path))
    print(f"number of {data_name} problems: {len(problems)}\n")
    return problems


class PintoDataset(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, tokenizer, source_len, target_len, args, diff_Pattern = False, OnlyAnswer = False
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
        def build_train_pair(problem, prompt_format):
            question = problem['words']
            question = ', '.join(question)
            answer = problem['answer']
            rationale = problem['explanation']

            input_format, output_format = prompt_format.split("-")
            if input_format=="Q":
                input = f"Words: {question}"
            elif input_format=="QE":
                input = f"Words: {question} \n Solution: {rationale}"
            elif input_format=="QA":
                input = f"Question: {question} \n Answer: The answer is {answer}."
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
        self.target_text = []
        self.source_text = []
        for problem in self.problems:
            if args.prompt_format=="Q[E]-A":
                prompt, target, pure_answer = build_train_pair(problem, "Q-A")
                self.target_text.append(target)
                self.source_text.append(prompt)
                prompt, target, pure_answer = build_train_pair(problem, "QE-A")
                self.target_text.append(target)
                self.source_text.append(prompt)
            else:
                prompt, target, pure_answer = build_train_pair(problem, args.prompt_format)
                if OnlyAnswer:
                    self.target_text.append(pure_answer)
                    self.source_text.append(prompt)
                else:
                    self.target_text.append(target)
                    self.source_text.append(prompt)
            
                

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

class Pipeline(nn.Module):
    def __init__(self,solution_giver, answer_extractor, tokenizer):
        super(Pipeline, self).__init__()
        self.solution_giver = solution_giver
        self.answer_extractor = answer_extractor
        self.tokenizer = tokenizer
    
    def generate(
                self,
                input_ids,
                attention_mask, 
                max_length=512, 
                num_beams=2
            ):
        input_decode = self.tokenizer.batch_decode(input_ids,skip_special_tokens=True, clean_up_tokenization_spaces=True)
        #print(input_decode)
        solution = self.solution_giver.generate(
                  input_ids = input_ids,
                  attention_mask = attention_mask, 
                  max_length = max_length, 
                  num_beams = num_beams
        )
        solution_decode = self.tokenizer.batch_decode(solution,skip_special_tokens=True, clean_up_tokenization_spaces=True)
        #print(solution_decode)
        after_cat = [(x.replace("Solution:","")+y)+"\nAnswer:" for (x,y) in zip(input_decode,solution_decode)]
        #print(after_cat)
        inputs = self.tokenizer.batch_encode_plus(
            after_cat,
            max_length=max_length,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids_2 = inputs["input_ids"].squeeze().to(device)
        attention_mask_2 = inputs["attention_mask"].squeeze().to(device)
        if len(input_ids_2.shape)==1:
            input_ids_2 = torch.stack([input_ids_2])
            attention_mask_2 = torch.stack([attention_mask_2])
        answer = self.answer_extractor.generate(
                input_ids = input_ids_2,
                attention_mask = attention_mask_2, 
                max_length = max_length, 
                num_beams = num_beams
        )
        return answer, solution_decode
        
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
    
    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenizer for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    def model_init():
        #return AutoModelForSeq2SeqLM.from_pretrained(model_params["MODEL"]) 
        if args.use_pipeline:
            answer_extractor = AutoModelForSeq2SeqLM.from_pretrained(model_params["MODEL"]) 
            answer_extractor.load_state_dict(torch.load(args.answer_extractor_path))
            solution_giver = AutoModelForSeq2SeqLM.from_pretrained(model_params["MODEL"]) 
            solution_giver.load_state_dict(torch.load(args.solution_giver_path))
            model = Pipeline(solution_giver,answer_extractor, tokenizer)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_params["MODEL"]) 
            model.load_state_dict(torch.load(args.answer_extractor_path))
        return model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")
    # unfold data frame

    train_problems = load_data("train",args) 
    test_problems = load_data("test",args) 
    dev_problems = load_data("dev",args) 

    # Creating the Training and Validation dataset for further creation of Dataloader
    train_set = PintoDataset(
        train_problems,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        args,
        #diff_Pattern = True
    )
    eval_set = PintoDataset(
        dev_problems,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        args,
        OnlyAnswer = True,
        #diff_Pattern = True
    )
    test_set = PintoDataset(
        test_problems,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        args,
        OnlyAnswer = True,
        #diff_Pattern = True
    )
    # Defining the parameters for creation of dataloaders
    
    model = model_init()
    
    def run(run_set,run_problems,prefix):
        correct = 0
        cnt = 0
        wrong_case = []
        right_case = []
        for _, pair in tqdm(enumerate(run_set)):
            problem = run_problems[cnt]
            cnt += 1
            prompt = tokenizer.decode(pair["input_ids"],skip_special_tokens=True, clean_up_tokenization_spaces=True)
            #print(prompt)
            ids = torch.stack([pair['input_ids']]).to(device)
            mask = torch.stack([pair['attention_mask']]).to(device)
            ref = torch.tensor(pair["labels"]).to(device)
            if args.use_pipeline:
                pred, sol = model.generate(
                    input_ids = ids,
                    attention_mask = mask, 
                    max_length=512, 
                    num_beams=1
                )
            else:
                pred = model.generate(
                    input_ids = ids,
                    attention_mask = mask, 
                    max_length=512, 
                    num_beams=2
                )
                sol = [""]
            pred = pred[0]
            sol = sol[0]
            #print(pred)
            
            ''' check correct '''
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
            pred_decode = tokenizer.decode(pred,skip_special_tokens=True, clean_up_tokenization_spaces=True)
            ref_decode = tokenizer.decode(ref,skip_special_tokens=True, clean_up_tokenization_spaces=True)
            gt_sol = problem["explanation"]
            question = ', '.join(problem['words'])
            if matched:
                correct = correct+ 1
                right_case_instance = {'num':_, 'question':question,'prediction solution':sol, 'ground truth solution':gt_sol, 'prediction':pred_decode,'reference':ref_decode}
                right_case.append(right_case_instance)
                case_instance = right_case_instance
            else:
                wrong_case_instance = {'num':_, 'question':question,'prediction solution':sol, 'ground truth solution':gt_sol, 'prediction':pred_decode,'reference':ref_decode}
                wrong_case.append(wrong_case_instance)
                case_instance = wrong_case_instance
            if _%100==0:
                print(_,"finished with correct:",correct/cnt)
                print(case_instance)
                correct_log = open(log_path,'a')
                correct_log.write(str(_)+" finished with correct: "+str(correct/cnt)+'\n')
                correct_log.close()

        print(correct)
        print("correct:",correct/cnt) 
        correct_log = open(log_path,'a')
        correct_log.write(str(cnt)+" finished with correct: "+str(correct/cnt)+'\n')
        correct_log.close()
        
        with open(f"{prefix}_wrong_cases.json","w") as os:
            json.dump(wrong_case,os)
        with open(f"{prefix}_right_cases.json","w") as os:
            json.dump(right_case,os)
    
    solution_giver_part = args.solution_giver_path.split('/')[-2]
    answer_extractor_part = args.answer_extractor_path.split('/')[-2]
    dir = args.answer_extractor_path.replace('pytorch_model.bin',f'cases_{solution_giver_part}')
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    if args.do_eval:
        prefix = args.answer_extractor_path.replace('pytorch_model.bin',f'cases_{solution_giver_part}/eval')
        print(prefix)
        correct_log = open(log_path,'a')
        correct_log.write('========================= eval =========================')
        correct_log.close()
        run(eval_set,dev_problems,prefix)
    
    if args.do_train:
        prefix = args.answer_extractor_path.replace('pytorch_model.bin',f'cases_{solution_giver_part}/train')
        print(prefix)
        correct_log = open(log_path,'a')
        correct_log.write('========================= train =========================')
        correct_log.close()
        run(train_set,train_problems,prefix)

    if args.do_test:
        prefix = args.answer_extractor_path.replace('pytorch_model.bin',f'cases_{solution_giver_part}/test_large_{args.use_pipeline}')
        print(prefix)
        correct_log = open(log_path,'a')
        correct_log.write('========================= test =========================')
        correct_log.close()
        run(test_set,test_problems,prefix)



if __name__ == '__main__':
    # let's define model parameters specific to T5
    model_params = {
        "MODEL": 'allenai/unifiedqa-t5-base',  # model_type: t5-base/t5-large
        "TRAIN_BATCH_SIZE": 4,  # training batch size
        "VALID_BATCH_SIZE": 8,  # validation batch size
        "TEST_BATCH_SIZE": 8,  # validation batch size
        "TRAIN_EPOCHS": 20,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "TEST_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 1e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 512,  # max length of target text
        "SEED": 42,  # set seed for reproducibility
    }
    
    T5Trainer(
        args = args,
        model_params=model_params,
        output_dir="outputs",
    )

