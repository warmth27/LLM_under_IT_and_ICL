# csv to json
import csv
import json


input_csv_file = 'data/bragging/test.csv'
output_jsonl_file = 'data/bragging/test_pro.jsonl'

# All potential categories
categories = ["bragging", "not bragging"]

# Open the input CSV file
with open(input_csv_file, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    # Open the output JSONL file
    with open(output_jsonl_file, mode='w', encoding='utf-8') as jsonl_file:
        for row in csv_reader:
            # Prepare the JSON object
            json_object = {
                "text": row['text'],
                "category": categories,
                "output": row['label']
                # "output": "-"
            }
            
            # Write the JSON object to the JSONL file
            jsonl_file.write(json.dumps(json_object, ensure_ascii=False) + '\n')
