import pandas as pd
import requests
import json
from tqdm import tqdm

# make sure the inference endpoint is running before executing this script

test = pd.read_excel("data/AI1000.xlsx")
submission = []
for idx, row in tqdm(test.iterrows(), total=test.shape[0]):

    prompt = "".join(
        [
            f"文章：{str(row['文章'])}\n\n",
            f"問題：{str(row['問題'])}\n\n",
            f"選項1：{str(row['選項1'])}\n"
            f"選項2：{str(row['選項2'])}\n"
            f"選項3：{str(row['選項3'])}\n"
            f"選項4：{str(row['選項4'])}\n",
        ]
    )

    data = {
        "model": "",
        "messages": [
            {
                "role": "system",
                "content": "接下來的訊息中，將會提供「文章」、「題目」，以及四個「選項」。請在仔細地閱讀並理解文章後，針對題目所述，從四個選項中選出最適當的答案。直接回答選項編號。",
            },
            {"role": "user", "content": prompt},
        ],
    }

    resp = requests.post(
        url="http://0.0.0.0:8000/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": "Bearer "},
        data=json.dumps(data),
    )
    resp = resp.json()
    submission.append(
        {"ID": int(row["題號"]), "Answer": resp["choices"][0]["message"]["content"]}
    )

submission = pd.DataFrame(submission)
submission.to_csv("submission.csv", index=False)
print(":: submission.csv created.")
