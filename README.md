# Natural Language Induced Adversarial Images
This is the official implementation repository for the paper "Natural Language Induced Adversarial Images" in ACM MM 2024.
## Installation
- Create a conda virtual environment and activate it:
```
conda create -n natural python=3.9 -y
conda activate natural
```
- install torch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- install required libraries in requirments.txt
```
pip install -r requirements.txt
```
- Set up Midjourney api
```
docker run -d --name midjourney-proxy  -p 8080:8080  -e mj.discord.guild-id=xxx  -e mj.discord.channel-id=xxx  -e mj.discord.user-token=xxx --restart=always  novicezk/midjourney-proxy:.6.3
```
You can find detailed instructions here: https://github.com/novicezk/midjourney-proxy

## Run the code
```
python search.py --algo gene  --target  dog --task animal --targetlabel 0 --selectclass 1 2 3 4 5 --target_score 20 --max_generations 50
```
The given command line parameters contain a few minor grammatical errors. Here is the corrected version:

We use three command line parameters:
- --algo: Used to specify the optimization algorithm you want to use. You can select from `gene` (Genetic Algorithm), `comb` (Combination Test), or `random` (Randomly select prompts).
- --target: The target category you want to attack.
- --task: The classification task you want to attack.
- --selectclass: Specify all the category indices in the ImageNet dataset.
- --targetlabel: The target category index in all the selected classes.
- --target_score: The target fitness score of the adversarial example.
- --max_generations: Maximum generation rounds.
