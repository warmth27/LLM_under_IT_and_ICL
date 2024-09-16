#! model = Qwen2-7b-Instruct
#! Lora ftuning

import json
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoTokenizer
import os
import random
import argparse



parser = argparse.ArgumentParser(description="Run experiments with different seeds and num_class values.")
parser.add_argument('--seed', type=int, required=True, help='Random seed')
parser.add_argument('--num_class', type=int, required=True, help='Number of examples per class')
args = parser.parse_args()


random_seed = args.seed
random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

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

def process_func(example):

    MAX_LENGTH = 512 
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|system|>\nAnalyze the content and determine if it includes {label}.Respond only with {label} or not {label}, without providing any additional context or explanation.<|endoftext|>\n<|user|>\n{example['input']}<|endoftext|>\n<|assistant|>\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}   


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=60)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


model_path = "/media/qust521/92afc6a9-13fd-4458-8a46-4b008127de08/LLMs/Qwen2-7b-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.enable_input_require_grads()  

train_dataset_path = "/home/qust521/Public/InstructionTuning/qwen2-7b/data/bragging/train_pro.jsonl"
test_dataset_path = "/home/qust521/Public/InstructionTuning/qwen2-7b/data/bragging/test_pro_nolabel.jsonl"
train_jsonl_new_path = "/home/qust521/Public/InstructionTuning/qwen2-7b/data/bragging/new_train_pro.jsonl"
test_jsonl_new_path = "/home/qust521/Public/InstructionTuning/qwen2-7b/data/bragging/new_test_pro_nolabel.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

with open(train_jsonl_new_path, "r") as file:
    train_examples = [json.loads(line) for line in file]

num_class = args.num_class
bragging_examples = [ex for ex in train_examples if "label: bragging" in ex['output']]
not_bragging_examples = [ex for ex in train_examples if 'not bragging' in ex['output']]
selected_examples = random.sample(bragging_examples, num_class) + random.sample(not_bragging_examples, num_class)

with open(f"/home/qust521/Public/InstructionTuning/qwen2-7b/samples/qwen2_7b_bragging_icl_{num_class}_{random_seed}.txt", "w", encoding="utf-8") as f:
    for example in selected_examples:
        f.write(example['input'] + example['output'] + "\n")

selected_df = pd.DataFrame(selected_examples)
selected_ds = Dataset.from_pandas(selected_df)
train_dataset = selected_ds.map(process_func, remove_columns=selected_ds.column_names)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    inference_mode=False,  
    r=8,  
    lora_alpha=32,  
    lora_dropout=0.1,  
)

model = get_peft_model(model, config)


args = TrainingArguments(
    output_dir=f"/home/qust521/Public/InstructionTuning/qwen2-7b/output/Qwen2-7b-Instruct-{num_class}-{random_seed}",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=2000,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)



trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

trainer.train()


test_df = pd.read_json(test_jsonl_new_path, lines=True)

instruction = "Analyze the content and determine if it includes {label}.Respond only with {label} or not {label}, without providing any additional context or explanation."

print("=================start to predict================")

results = []

test_text_list = []
for index, row in test_df.iterrows():
    instruction = instruction
    input_value = row['input']
    
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)

    prediction = {
        "input": row["input"],
        "prediction": response,
    }
    results.append(prediction)
    messages.append({"role": "assistant", "content": f"{response}"})
    result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"

with open(f"/home/qust521/Public/InstructionTuning/qwen2-7b/qwen2_7b_bragging_icl_{num_class}_{random_seed}.json", "w", encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
