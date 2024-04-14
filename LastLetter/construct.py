import os
import random
import json
# Initialize an empty list to store the lines
random.seed(42)
words = []
order_words = ['first', 'second', 'third', 'forth', 'fifth', 'sixth', 'seventh', 'eighth','nineth','tenth']

# Open the file for reading
with open("./google-10000-english-no-swears.txt", 'r') as file:
    # Read each line and add it to the list
    for line in file:
        words.append(line.strip())  # Remove trailing newline characters
# for dev/test split, left_bound = 8000 and right_bound = len(words)-1 
# for train split, left_bound = 0 and right_bound = 8000
left_bound = 8000 
right_bound = len(words)-1 
output_file = "dev.json"
res = []
max_len = 5
sample_times = 1000


def sample_word(l=0, r=8000):
    ran = random.randint(l,r)
    return words[ran]
for l in range(max_len):
    for j in range(sample_times):
        sampled_words = []
        for i in range(l+1):
            sampled_words.append(sample_word(left_bound,right_bound))
        explanation =' '.join([f"The last letter of the {order_word} word '{word}' is '{word[-1]}'." for word, order_word in zip(sampled_words, order_words)])
        answer = ''.join([word[-1] for word in sampled_words])
        res_ = {}
        res_['words'] = sampled_words
        res_['explanation'] = explanation
        res_['answer'] = answer
        
        res.append(res_)

with open(output_file,"w") as os:
    json.dump(res,os)
