# traditional-chinese-reading-comprehension-test-for-llms

This is the repo for storing scripts, configuration and results of the second Kaggle inClass competition of NYCU-IAIS-DL2024. The goal is to train a large language model to comprehend articles written in Traditional Chinese and respond accurately to associated questions.

In this project, we use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), a LLM Fine-Tuning Toolbox, to fine-tune LLM with the given reading materials.

## 

driver: 535.171.04
cuda: 12.2

NVIDIA GeForce RTX 4080

## Setup environment

```
# create project directory
mkdir competition-2
cd competition-2
# clone llama-factory and install dependencies
virtualenv .llama-factory
source .llama-factory/bin/activate
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.0 triton \
  --index-url https://download.pytorch.org/whl/cu121
pip install packaging
pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"
pip install -e .[bitsandbytes]
pip install openpyxl # for process train data
```

## Download and prepare the train data

```
# back to project root
cd ..
kaggle competitions download -c zh-tw-reading-comprehension-test-for-llms
unzip zh-tw-reading-comprehension-test-for-llms.zip -d data/
python data_prep.py
```

This will convert the dataset file by the required **alpaca** format and save as `zh_tw_reading_comprehension.json` under `LLaMA-Factory/data`, then add the dataset description to `dataset_info.json`.

## Train

Run `train.sh` under project root will begin the training.

## Inference

Running `inference.sh` script will initiate the LLaMA-Factory inference API endpoint. Then, the script will execute inferences using test data via the endpoint and subsequently close the endpoint upon completion. Finally, it will generate `submission.csv` file containing the results.

## Experiments

1. LLaMA3-8B/sft1: initial
2. LLaMA3-8B/sft2: use improved prompt, unsloth, lora rank=16
3. LLaMA3-8B/sft3: use improved prompt, lora rank=16
4. Breeze-7B/sft1: use improved prompt, lora rank=16
5. LLaMA3-8B/sft4: use unsloth, zh_tw_reanding_comprehension + alpaca_gpt4_zh

## Results

1. First submission: 0.88333
2. Use unsloth, structured prompt, rank=16: 0.86333
3. Use structured prompt, rank=16: 0.89333
4. very slow inference, need quantization down to 4 bits: error 
5. double fine tune time: 0.90666
