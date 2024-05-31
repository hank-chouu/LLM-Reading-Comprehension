import pandas as pd
import json
import os
from tqdm import tqdm

# process train

train = pd.read_excel("data/AI.xlsx")
prompts = []
print(":: Data conversion started.")

for _, row in tqdm(train.iterrows(), total=train.shape[0]):
    instruction = "".join(
        [
            "接下來的訊息中，將會提供「文章」、「題目」，以及四個「選項」。請在仔細地閱讀並理解文章後，針對題目所述，從四個選項中選出最適當的答案。直接回答選項編號。\n\n",
            f"文章：{str(row['文章'])}\n\n",
            f"問題：{str(row['問題'])}\n\n",
            f"選項1：{str(row['選項1'])}\n"
            f"選項2：{str(row['選項2'])}\n"
            f"選項3：{str(row['選項3'])}\n"
            f"選項4：{str(row['選項4'])}\n",
        ]
    )
    output = f"{str(row['正確答案'])}"
    prompt = {"instruction": instruction, "input": "", "output": output}
    prompts.append(prompt)

# Save the processed data into a JSON file
data_dir = "LLaMA-Factory/data"
data_name = "zh_tw_reading_comprehension"
file_name = data_name + ".json"
with open(os.path.join(data_dir, file_name), "w", encoding="utf-8") as f:
    json.dump(prompts, f, indent=2, ensure_ascii=False)

print(":: zh_tw_reading_comprehension.json saved successfully.")

# modify dataset_info.json

with open(os.path.join(data_dir, "dataset_info.json"), "r") as f:
    dataset_info = json.load(f)
if data_name not in dataset_info.keys():
    dataset_info[data_name] = {"file_name": file_name, "formatting": "alpaca"}
    with open(os.path.join(data_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(":: dataset_info.json modified!")
else:
    print(":: dataset_info.json already contains zh_tw_reading_comprehension.")
