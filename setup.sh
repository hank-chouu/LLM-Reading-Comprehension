#!/bin/bash
set -e
pwd=$PWD
echo ":: Setup started"
echo ":: Creating project folder (../zh-tw-reading-comprehension-test-for-llms)"
rm -rf ../zh-tw-reading-comprehension-test-for-llms
mkdir ../zh-tw-reading-comprehension-test-for-llms
echo ":: Copying scripts"
cp scripts/* ../zh-tw-reading-comprehension-test-for-llms
cd ../zh-tw-reading-comprehension-test-for-llms
# clone llama-factory and install dependencies
echo ":: Creating virtual enviromnent (.llama-factory)"
python3 -m venv .llama-factory
source .llama-factory/bin/activate
pip install --upgrade pip
pip install setuptools wheel
echo ":: Installing LLaMA-Factory and dependencies"
git clone https://github.com/hiyouga/LLaMA-Factory.git
pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.0 triton \
  --index-url https://download.pytorch.org/whl/cu121
pip install packaging
pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"
pip install -e LLaMA-Factory[bitsandbytes]
pip install openpyxl pandas tqdm # for process train data
# download and unzip the training file into data dir
echo ":: Downloading data"
kaggle competitions download -c zh-tw-reading-comprehension-test-for-llms
unzip zh-tw-reading-comprehension-test-for-llms.zip -d data/
python data_prep.py
cd $pwd
echo ":: Setup complete"
