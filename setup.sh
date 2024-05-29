#!/bin/bash
set -e
pwd=$PWD
echo ":: Setup started"
echo ":: Creating project folder"
rm -rf ../zh-tw-reading-comprehension-test-for-llms
mkdir ../zh-tw-reading-comprehension-test-for-llms
echo ":: Copying scripts"
cp train.sh inference.sh data_prep.py generate_submission.py ../zh-tw-reading-comprehension-test-for-llms
cd ../zh-tw-reading-comprehension-test-for-llms
# clone llama-factory and install dependencies
echo ":: Creating virtual enviromnent (.llama-factory)"
virtualenv .llama-factory > /dev/null
source .llama-factory/bin/activate
echo ":: Installing LLaMA Factory"
git clone https://github.com/hiyouga/LLaMA-Factory.git
echo ":: Installing dependencies"
pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.0 triton \
  --index-url https://download.pytorch.org/whl/cu121
pip install packaging
pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"
cd LLaMA-Factory
pip install -e .[bitsandbytes]
pip install openpyxl # for process train data
# download and unzip the training file into data dir
echo ":: Downloading data..."
kaggle competitions download -c zh-tw-reading-comprehension-test-for-llms
unzip zh-tw-reading-comprehension-test-for-llms.zip -d data/
python data_prep.py
cd $pwd
echo ":: Setup complete"
