#! model = Phi-3-small-128k-instruct 
#! ICL

#! need to install
# MAX_JOBS=8 pip install flash-attn --no-build-isolation


import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import random
import argparse


parser = argparse.ArgumentParser(description="Run experiments with different seeds and num_class values.")
parser.add_argument('--seed', type=int, required=True, help='Random seed')
parser.add_argument('--num_class', type=int, required=True, help='Number of examples per class')
args = parser.parse_args()

def dataset_jsonl_transfer(origin_path, new_path):
    examples = []
    with open(origin_path, "r") as file:
        for line in file:
            data = json.loads(line)
            context = data["text"]
            label = data["output"]
            example = {
                "input": f"tweet: {context}\n",
                "output": f"label: {label} \n",
            }
            examples.append(example)

    with open(new_path, "w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

def generate_prompt(examples):
    prompt = ""
    for example in examples:
        prompt += example['input'] + example['output']
    return prompt

def predict(prompt, model, tokenizer, new_input):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    temperature = 0.2

    prompt += "Analyze the content and determine if it includes {label}.Respond only with {label} or not {label}, without providing any additional context or explanation." 

    new_input = f"{new_input}"

    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": new_input}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, 
                                   max_new_tokens=50, 
                                   temperature = temperature,
                                   eos_token_id=tokenizer.encode('<|endoftext|>')[0])
    
    outputs = generated_ids.tolist()[0][len(model_inputs[0]):]
    response = tokenizer.decode(outputs).split('<|end|>')[0]

    return response


model_path = "/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/LLMs/Phi-3-small-128k-instruct "
train_dataset_path = "/home/qust521/Public/InContextLearning/Qwen2-7b/data/bragging/train_pro.jsonl"
test_dataset_path = "/home/qust521/Public/InContextLearning/Qwen2-7b/data/bragging/test_pro_nolabel.jsonl"
train_jsonl_new_path = "/home/qust521/Public/InContextLearning/Qwen2-7b/data/bragging/new_train_pro.jsonl"
test_jsonl_new_path = "/home/qust521/Public/InContextLearning/Qwen2-7b/data/bragging/new_test_pro_nolabel.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": "cuda"}, torch_dtype=torch.bfloat16, trust_remote_code=True)

with open(train_jsonl_new_path, "r") as file:
    train_examples = [json.loads(line) for line in file]


random_seed = args.seed
random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)


bragging_examples = [ex for ex in train_examples if "label: bragging" in ex['output']]
not_bragging_examples = [ex for ex in train_examples if 'not bragging' in ex['output']]

num_class = args.num_class
selected_examples = random.sample(bragging_examples, num_class) + random.sample(not_bragging_examples, num_class)


with open(f"/home/qust521/Public/InContextLearning/Qwen2-7b/samples/qwen2_7b_bragging_icl_{num_class}_{random_seed}.txt", "w", encoding="utf-8") as f:
    for example in selected_examples:
        f.write(example['input'] + example['output'] + "\n")

test_df = pd.read_json(test_jsonl_new_path, lines=True)
results = []


for index, row in test_df.iterrows():
    new_input = row['input']
    
    prompt = generate_prompt(selected_examples) 

    response = predict(prompt, model, tokenizer, new_input)

   
    predictions = {
        "input": row["input"],
        "prediction": response,
    }
    results.append(predictions)

with open(f"/home/qust521/Public/InContextLearning/Qwen2-7b/qwen2_7b_bragging_icl_{num_class}_{random_seed}.json", "w", encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
