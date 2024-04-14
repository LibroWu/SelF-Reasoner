import os
import numpy as np
import pandas as pd
import json

basepath = ""
path_pipeline_right = f"{basepath}/test_right_cases.json"
path_pipeline_wrong = f"{basepath}/test_wrong_cases.json"
path_direct_right = f"{basepath}/test_direct_right_cases.json"
path_direct_wrong = f"{basepath}/test_direct_wrong_cases.json"

with open(path_direct_right, 'r') as fcc_file:
    cases_direct_right = json.load(fcc_file)
num_set_direct_right = {}
for case in cases_direct_right:
    num_set_direct_right[case['num']] = case

with open(path_direct_wrong, 'r') as fcc_file:
    cases_direct_wrong = json.load(fcc_file)
num_set_direct_wrong = {}
for case in cases_direct_wrong:
    num_set_direct_wrong[case['num']] = case

with open(path_pipeline_right, 'r') as fcc_file:
    cases_pipeline_right = json.load(fcc_file)
num_set_pipeline_right = {}
for case in cases_pipeline_right:
    num_set_pipeline_right[case['num']] = case

with open(path_pipeline_wrong, 'r') as fcc_file:
    cases_pipeline_wrong = json.load(fcc_file)
num_set_pipeline_wrong = {}
for case in cases_pipeline_wrong:
    num_set_pipeline_wrong[case['num']] = case
count = 0
cnt = 0
pipeline_chosen = 0
for i in range(1000):
    solved_by_direct = i in num_set_direct_right
    solved_by_pipeline = i in num_set_pipeline_right
    if solved_by_direct:
        direct_res = num_set_direct_right[i]
    else:
        direct_res = num_set_direct_wrong[i]
    if solved_by_pipeline:
        pipeline_res = num_set_pipeline_right[i]
    else:
        pipeline_res = num_set_pipeline_wrong[i]
    question = pipeline_res['question'].split(', ')
    ref_answer = pipeline_res['reference']
    ref_exp = pipeline_res['ground truth solution']
    pred_exp = pipeline_res['prediction solution']
    pipeline_pred = pipeline_res['prediction']
    direct_pred = direct_res['prediction']
    flag = True
    for word in question:
        if not (f"'{word}'" in pred_exp):
            flag = False
            break
    pipeline_chosen += flag
    if flag:
        ans = pipeline_pred
    else:
        ans = direct_pred
    if ans == ref_answer:
        count += 1
    else:
        if solved_by_direct or solved_by_pipeline:
            cnt += 1
        
print(cnt)
print(count/1000)
print(pipeline_chosen)