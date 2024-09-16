from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import json
import pandas as pd
from datasets import Dataset
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import argparse
import random

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'


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

    messages = []

    with open(origin_path, "r") as file:
        for line in file:
            data = json.loads(line)
            context = data["text"]
            label = data["output"]
            message = {
                "instruction": "Identify whether a tweet is a customer complaint or a non-complaint. Only reply (complaint) or (not complaint).",
                "input": f"{context}",
                "output": label,
            }
            messages.append(message)

    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")
            
train_dataset_path = "/opt/data/private/ICL-IT/datasets/complaint/train_pro.json"
test_dataset_path = "/opt/data/private/ICL-IT/datasets/complaint/test_pro_nolable.json"
train_jsonl_new_path = "/opt/data/private/ICL-IT/datasets/complaint/new_train.jsonl"
test_jsonl_new_path = "/opt/data/private/ICL-IT/datasets/complaint/new_test_nolabel.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

with open(train_jsonl_new_path, "r") as file:
    data = [json.loads(line) for line in file]


num_class = args.num_class
bragging_examples = [ex for ex in data if "complaint" in ex['output']]
not_bragging_examples = [ex for ex in data if 'not complaint' in ex['output']]
selected_examples = random.sample(bragging_examples, num_class) + random.sample(not_bragging_examples, num_class)

with open(f"/opt/data/private/ICL-IT/IT/glm/complaint/sample/glm_it_{num_class}_{random_seed}.txt", "w", encoding="utf-8") as f:
    for example in selected_examples:
        f.write(example['input'] + example['output'] + "\n")
        

selected_df = pd.DataFrame(selected_examples)
ds = Dataset.from_pandas(selected_df)



tokenizer = AutoTokenizer.from_pretrained('/opt/data/private/ICL-IT/glm', use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer((f"[gMASK]<sop><|system|>\n Analyze the content and determine if it includes {label}.Respond only with {label} or not {label}, without providing any additional context or explanation. <|user|>\n"
                            f"{example['instruction']+example['input']}<|assistant|>\n"
                            ).strip(), 
                            add_special_tokens=False)
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1] 
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))


model = AutoModelForCausalLM.from_pretrained('/opt/data/private/ICL-IT/glm', device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True)
model.enable_input_require_grads() 



config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],  
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
args = TrainingArguments(
    output_dir=f"/opt/data/private/ICL-IT/IT/glm/complaint/output/GLM4-{num_class}-{random_seed}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=50,
    num_train_epochs=2,
    save_strategy="epoch",
    learning_rate=1e-5,
    save_on_each_node=True,
    gradient_checkpointing=True
)
trainer = Trainer(
    model=model.to(device), 
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
trainer.train()
peft_model_id="/opt/data/private/ICL-IT/IT/glm/complaint/output/GLM4_lora"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PeftModel.from_pretrained(model, model_id=peft_model_id).to(device)


output_file = f'/opt/data/private/ICL-IT/IT/glm/complaint/results/glm4_{num_class}_{random_seed}.json'  


results = []

with open(test_jsonl_new_path, 'r',encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip()) 
        tweet_text = item['input']
        prompt = item['instruction']

        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": tweet_text}, {"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(device)

        gen_kwargs = {"max_length": 250, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({"input": tweet_text, "prediction": prediction})

with open(output_file, 'w',encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"Predictions saved to {output_file}")


