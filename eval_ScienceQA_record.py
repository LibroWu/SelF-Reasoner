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
    parser.add_argument('--data_root', type=str, default='ScienceQA/data/scienceqa')
    parser.add_argument('--output_root', type=str, default='./results')
    parser.add_argument('--caption_file', type=str, default='ScienceQA/data/captions.json')
    parser.add_argument('--model_solution', type=str, default="allenai/unifiedqa-t5-base")
    parser.add_argument('--model_answer', type=str, default="allenai/unifiedqa-t5-base")
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    # user options
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--train', type=str, default='True', choices=['True','False'])
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival','test_no_image','val_no_image'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest','test_no_image'])
    parser.add_argument('--test_number', type=int, default=10, help='GPT-3 is expensive. -1 for whole val/test set')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--save_every', type=int, default=10, help='Save the result with every n examples.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--prompt_format',
                        type=str,
                        default='CQM-A',
                        choices=[
                            'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE', 'QCM-A',
                            'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE', 'QCML-A', 'QCME-A',
                            'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE','QCM-E'
                        ],
                        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=3, help='Number of n-shot training examples.')
    parser.add_argument('--shot_qids', type=list, default=None, help='Question indexes of shot examples')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='text-davinci-002')
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--num_beams', type=int, default=2)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--top_k', type=float, default=None)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--solution_giver_path', type=str, default='')
    parser.add_argument('--answer_extractor_path', type=str, default="")
    parser.add_argument('--frequency_penalty', type=float, default=None)
    parser.add_argument('--presence_penalty', type=float, default=None)
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
    
    log_path = args.answer_extractor_path.replace('pytorch_model.bin',f'correct_log_pipeline_{target}_{args.use_caption}_{timestamp}.txt')
    correct_log = open(log_path,'w')
    correct_log.write("args "+str(args)+'\n')
    correct_log.write('====Input Arguments====\n')
    correct_log.write(json.dumps(vars(args), indent=2, sort_keys=False)+'\n')
    correct_log.close()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
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
            prompt, target, pure_answer = build_train_pair(problems, [], qid, args)
            if OnlyAnswer:
                self.target_text.append(pure_answer)
                prompt.replace("Answer","Solution")
                self.source_text.append(prompt+"Let's think step by step!")
                #self.source_text.append(prompt)
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

'''
Input format: QCM Solution:
Pipeline I output format: Solution: [Solution]
Pipeline II input format: QCM BECAUSE: [Solution]
Pipeline II output format: Answer: The answer is [Answer]
'''
class PredictPipeline(nn.Module):
    def __init__(self, explain_generator, answer_giver,tokenizer, short_cut = False):
        super(PredictPipeline, self).__init__()
        self.explain_generator = explain_generator
        self.answer_giver = answer_giver
        self.tokenizer = tokenizer
        self.short_cut = short_cut
    
    # do not support batchsize>1 when apply self-consistency
    def generate(self, input_ids, attention_mask, max_length=512,num_beams=2, num_return_sequences=1, top_k=None, top_p = None, do_sample = False, temperature = 0.8):
        input_decode = self.tokenizer.batch_decode(input_ids,skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if self.short_cut:
            answer = self.answer_giver.generate(
                  input_ids = input_ids,
                  attention_mask = attention_mask, 
                  max_length = max_length, 
                  num_beams = num_beams
            )
            solution_decode = [""]
        else:
            # if do_sample then self-consistence
            if do_sample:
                beam_solution = self.explain_generator.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask, 
                    max_length = max_length, 
                    num_beams = num_beams,
                    num_return_sequences = num_return_sequences,
                    do_sample = do_sample,
                    top_k = top_k,
                    top_p = top_p,
                    temperature = temperature,
                )
                solution_decode = self.tokenizer.batch_decode(beam_solution,skip_special_tokens=True, clean_up_tokenization_spaces=True)
                print(solution_decode)
                after_cat = [(input_decode[0]+y).replace('Solution:Solution','\nBecause')+"\nAnswer:" for y in solution_decode]
                print('after_cat',after_cat)
                pipeline2_input = self.tokenizer.batch_encode_plus(
                    after_cat,
                    max_length=max_length,
                    pad_to_max_length=True,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids_2 = pipeline2_input["input_ids"].to(device)
                attention_mask_2 = pipeline2_input["attention_mask"].to(device)
                if len(input_ids_2.shape)==1:
                    input_ids_2 = torch.stack([input_ids_2])
                    attention_mask_2 = torch.stack([attention_mask_2])
                answer = self.answer_giver.generate(
                    input_ids = input_ids_2,
                    attention_mask = attention_mask_2, 
                    max_length = max_length, 
                    num_beams = num_beams
                )
                answer_decode = self.tokenizer.batch_decode(answer,skip_special_tokens=True, clean_up_tokenization_spaces=True)
                print(answer_decode)
                exit()
            else:
                solution = self.explain_generator.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask, 
                    max_length = max_length, 
                    num_beams = num_beams
                )
                solution_decode = self.tokenizer.batch_decode(solution,skip_special_tokens=True, clean_up_tokenization_spaces=True)
                after_cat = [(x+y).replace('Solution:Solution','\nBecause')+"\nAnswer:" for (x,y) in zip(input_decode,solution_decode)]
                pipeline2_input = self.tokenizer.batch_encode_plus(
                    after_cat,
                    max_length=max_length,
                    pad_to_max_length=True,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids_2 = pipeline2_input["input_ids"].squeeze().to(device)
                attention_mask_2 = pipeline2_input["attention_mask"].squeeze().to(device)
                if len(input_ids_2.shape)==1:
                    input_ids_2 = torch.stack([input_ids_2])
                    attention_mask_2 = torch.stack([attention_mask_2])
                answer = self.answer_giver.generate(
                    input_ids = input_ids_2,
                    attention_mask = attention_mask_2, 
                    max_length = max_length, 
                    num_beams = num_beams
                )
            return answer, solution_decode
    
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
    #tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    logits, labels = eval_preds
    eval_result = compute(predictions=logits, references=labels)
    correct_log = open(log_path,'a')
    correct_log.write(str(eval_result)+'\n')
    correct_log.close()
    print(eval_result)
    return eval_result

        
def T5Trainer(
    dataframe, args, model_params, output_dir="./outputs/"
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
        model = AutoModelForSeq2SeqLM.from_pretrained(model_params["MODEL"]) 
        model.load_state_dict(torch.load(args.answer_extractor_path))
        return model.to(device)

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
        OnlyAnswer = True,
    )
    
    test_set = ScienceQADataset(
        problems,
        test_qids,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        args,
        OnlyAnswer = True,
    )

    # Defining the parameters for creation of dataloaders
    
    model = model_init()

    
    
    def run(run_set,run_qids,prefix):
        correct = 0
        cnt = 0
        wrong_case = []
        right_case = []
        for _, pair in enumerate(run_set):
            cnt += 1
            prompt = tokenizer.decode(pair["input_ids"],skip_special_tokens=True, clean_up_tokenization_spaces=True)
            ids = torch.stack([pair['input_ids']]).to(device)
            mask = torch.stack([pair['attention_mask']]).to(device)
            ref = torch.tensor(pair["labels"]).to(device)
            pred = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=512, 
                num_beams=2
            )
            pred = pred[0]
            sol = ""
            
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
            gt_sol = problems[run_qids[_]]['solution']
            question = problems[run_qids[_]]['question']
            options = problems[run_qids[_]]['choices']
            if matched:
                correct = correct+ 1
                right_case_instance = {'num':_, 'question':question,'choices':options,'prediction solution':sol, 'ground truth solution':gt_sol, 'prediction':pred_decode,'reference':ref_decode}
                right_case.append(right_case_instance)
                case_instance = right_case_instance
            else:
                wrong_case_instance = {'num':_, 'question':question,'choices':options,'prediction solution':sol, 'ground truth solution':gt_sol, 'prediction':pred_decode,'reference':ref_decode}
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
            
    answer_extractor_part = args.answer_extractor_path.split('/')[-2]
    dir = args.answer_extractor_path.replace('pytorch_model.bin',f'cases_{answer_extractor_part}_{args.use_caption}')
    if not os.path.exists(dir):
        os.makedirs(dir)
    if args.do_test:
        #prefix = f"unifiedq-t5-base_test_{args.solution_giver_path.split('/')[-1]}_append_use_caption"
        prefix = args.answer_extractor_path.replace('pytorch_model.bin',f'cases_{answer_extractor_part}_{args.use_caption}/test')
        print(prefix)
        correct_log = open(log_path,'a')
        correct_log.write('========================= test =========================')
        correct_log.close()
        run(test_set,test_qids,prefix)
    
    if args.do_eval:
        #prefix = f"unifiedq-t5-base_test_{args.solution_giver_path.split('/')[-1]}_append_use_caption"
        prefix = args.answer_extractor_path.replace('pytorch_model.bin',f'cases_{answer_extractor_part}_{args.use_caption}/eval')
        print(prefix)
        correct_log = open(log_path,'a')
        correct_log.write('========================= eval =========================')
        correct_log.close()
        run(eval_set,val_qids,prefix)
    
    if args.do_train:
        #prefix = f"unifiedq-t5-base_test_{args.solution_giver_path.split('/')[-1]}_append_use_caption"
        prefix = args.answer_extractor_path.replace('pytorch_model.bin',f'cases_{answer_extractor_part}_{args.use_caption}/train')
        print(prefix)
        correct_log = open(log_path,'a')
        correct_log.write('========================= train =========================')
        correct_log.close()
        run(train_set,train_qids,prefix)



if __name__ == '__main__':
    problems, qids, shot_qids = load_data(args)  # probelms, test question ids, shot example ids
    dataframe = {'problems':problems, 'qids':qids, 'shot_qids':shot_qids}


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
        dataframe=dataframe,
        args = args,
        model_params=model_params,
        output_dir="outputs",
    )
