# traditional-chinese-reading-comprehension-test-for-llms

This is the report of the second Kaggle inClass competition of NYCU-IAIS-DL2024, by Heng-Tse Chou (NTHU STAT). The goal is to train a large language model to comprehend articles written in Traditional Chinese and respond accurately to associated questions.

In this project, we use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), a LLM Fine-Tuning Toolbox, to fine-tune LLM with the given reading materials.

## Environment

- OS information: Ubuntu 22.04 LTS
- Python version: 3.10.12
- GPU: NVIDIA GeForce RTX 4080
- Driver version: 535.171.04
- CUDA version: 12.2

## Setup

To reproduce the project setup, simply run `setup.sh` under this repo.

The script will do the following sequentially:

1. Create the project directory `zh-tw-reading-comprehension-test-for-llms` under the parent directory.
2. Copy the reqired scripts into the project directory.
3. Create and activate a virtual environment (.llama-factory).
4. Clone LLaMA-Factory and install the dependencies.
5. Download and unzip the questions and the reading materials.
6. Convert the questions and the reading materials into `zh_tw_reading_comprehension.json`, a dataset of instructions (See the data prep part) under `LLaMA-Factory/data`.
7. Add new dataset description into `dataset_info.json`.

## Data Prep

The original data are tables with columns { 文章, 問題, 選項 1, 選項 2, 選項 3, 選項 4, 正確答案 }. Here, we follow the **Alpaca** format and construct the training dataset.

We have two versions of the training instructions. We found that ver.2, which provides clearer and more concrete instructions, leads to improved inference results.

Ver. 1

```python
instruction = "".join([
    "接下來的訊息中，將會提供文章、題目，以及選項。請在仔細的閱讀並理解文章後，針對題目所述，從四個選項中選出最適當的答案。直接回答選項編號。\n\n",
    f"文章：{str(row['文章'])}\n\n",
    f"問題：{str(row['問題'])}\n\n",
    f"選項1：{str(row['選項1'])}\n"
    f"選項2：{str(row['選項2'])}\n"
    f"選項3：{str(row['選項3'])}\n"
    f"選項4：{str(row['選項4'])}\n"
])
output = f"{str(row['正確答案'])}"
prompt = {"instruction": instruction, "input": "", "output": output}
```

Ver. 2

```python
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
```

## Train

Run `train.sh` under project root will begin the training. The initial settings are given below.

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path unsloth/llama-3-8b-Instruct-bnb-4bit \
    --template llama3 \
    --dataset_dir data \
    --dataset zh_tw_reading_comprehension \
    --finetuning_type lora \
    --lora_target all \
    --loraplus_lr_ratio 16.0 \
    --quantization_bit 4 \
    --upcast_layernorm True \
    --cutoff_len 1024 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_ratio 0.1 \
    --optim adamw_torch \
    --output_dir saves/LLaMA3-8B/sft \
    --overwrite_output_dir \
    --plot_loss \
    --fp16 True
```

We basically specify that

- Use the `llama-3-8b-Instruct-bnb-4bit` model from unsloth (The unsloth acceleration tool is not used in th initial settings though).
- Template: llama3
- Use `lora_target: all` to specify all the available modules for better convergence.
- Loraplus LR ratio = 16
- Learning rate = 5e-5
- Batch = 3
- Gradient Accumulation Steps = 8
- Optimizer = AdamW
- Warmup ratio = 0.1

In different Self-supervised Fine-Tune experiments, only a small fraction of settings are changed, which will be elaborated in the experiments part.

## Inference

Running `inference.sh` script will initiate the LLaMA-Factory inference API endpoint. Then, the script will execute inferences using test data via the endpoint and subsequently close the endpoint upon completion. Finally, it will generate `submission.csv` file containing the results.

The inference prompt is given as follow, and the two versions of prompts for inference are in correspondence of the two versions of prompts in data prep.

Ver. 1

```python
prompt = "".join([
    f"文章：{str(row['文章'])}\n\n",
    f"問題：{str(row['問題'])}\n\n",
    f"選項1：{str(row['選項1'])}\n"
    f"選項2：{str(row['選項2'])}\n"
    f"選項3：{str(row['選項3'])}\n"
    f"選項4：{str(row['選項4'])}\n"
])

data={
    "model": "",
    "messages": [
        {
            "role": "system",
            "content": "接下來的訊息中，將會提供文章、題目，以及選項。請在仔細的閱讀並理解文章後，針對題目所述，從四個選項中選出最適當的答案。直接回答選項編號。"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
}
```

Ver. 2

```python
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
        {
            "role": "user",
            "content": prompt
        },
    ],
}
```

## Experiments

### LLaMA3-8B/sft1

We adopted the initial settings and used prompt version 1, completing fine-tuning in approximately 6 hours.

<p align="center">
  <img src="https://github.com/Deep-Learning-NYCU/traditional-chinese-reading-comprehension-test-for-llms-A112092-new/blob/main/img/llama3-8b-sft1.png?raw=true" alt="LLaMA3-8B/sft1"/>
</p>

- Public score: 0.88333
- Private score: 0.89571

### LLaMA3-8B/sft2

In this experiment, we used prompt version 2, adopted the unsloth acceleration, and set the LORA rank to 16. The fine-tuning process lasted for approximately 3.5 hours.

<p align="center">
  <img src="https://github.com/Deep-Learning-NYCU/traditional-chinese-reading-comprehension-test-for-llms-A112092-new/blob/main/img/llama3-8b-sft2.png?raw=true" alt="LLaMA3-8B/sft2"/>
</p>

- Public score: 0.86333
- Private score: 0.88714

Despite employing a more concrete prompt and a higher LORA rank, a decrease in accuracy was observed.

### LLaMA3-8B/sft3

This experiment replicated the settings of sft2 but without using the unsloth part intended for speed-up.

<p align="center">
  <img src="https://github.com/Deep-Learning-NYCU/traditional-chinese-reading-comprehension-test-for-llms-A112092-new/blob/main/img/llama3-8b-sft3.png?raw=true" alt="LLaMA3-8B/sft3"/>
</p>

- Public score: 0.89333
- Private score: 0.88857

The score in this experiment was higher than those observed in sft1 and sft2, suggesting that using unsloth might actually decrease accuracy.

### Breeze-7B/sft1

In this experiment, we tried a different model, but it did not work out. The training and inference each took 6 hours, and the submission created was garbled.

<p align="center">
  <img src="https://github.com/Deep-Learning-NYCU/traditional-chinese-reading-comprehension-test-for-llms-A112092-new/blob/main/img/llama3-8b-sft1.png?raw=true" alt="Breeze-7B/sft1"/>
</p>

It is possible that an incorrect template or settings were selected, but this matter was not investigated further.

### LLaMA3-8B/sft4

In this experiment, we examined whether providing the model with additional traditional Chinese material could enhance its reading comprehension abilities and lead to higher scores on comprehension tests.

The configuration was identical to sft2, except we added the `alpaca_gpt4_zh` dataset as additional training material. The fine-tuning process took about 12 hours.

<p align="center">
  <img src="https://github.com/Deep-Learning-NYCU/traditional-chinese-reading-comprehension-test-for-llms-A112092-new/blob/main/img/llama3-8b-sft4.png?raw=true" alt="LLaMA3-8B/sft4"/>
</p>

- Public score: 0.90666
- Private score: 0.89428

This was the best result we obtained.

### LLaMA3-8B/sft5

In this experiment, we aimed to investigate whether excluding unsloth could improve accuracy. The configuration was identical to sft4, but we did not use unsloth. The fine-tuning process took about 25 hours.

<p align="center">
  <img src="https://github.com/Deep-Learning-NYCU/traditional-chinese-reading-comprehension-test-for-llms-A112092-new/blob/main/img/llama3-8b-sft5.png?raw=true" alt="LLaMA3-8B/sft5"/>
</p>

- Public score: 0.86666
- Private score: 0.87571

It was surprising to see that the result was worse than sft1.

## Conclusion

Nowadays, fine-tuning a LLM is a straightforward process. While most available LLMs are intelligent, the results of fine-tuning are consistently positive. However, it can sometimes be challenging to enhance fine-tuning further, as the precise factors contributing to differences remain uncertain. Moreover, it can be difficult to determine whether the model is truly improving.

In our experiments, we discovered several factors that might improve the model:

1. Better prompt.
2. Higher LORA rank, as it will retain more information (in exchnage of the computational cost).
3. More training data.

On the other side, several factors might leads to a decreased performance but we are not sure:

1. Using an acceleration tool such as unsloth.
