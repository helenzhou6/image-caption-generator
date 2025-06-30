# Image Caption Generator
Aim: To take an image and generate a caption with **magic** (a.k.a a CLIP Visual Encoder + Custom Decoder)
MLX week 4.

## Pre-requisities 
- Python 3.10
- uv - install via https://github.com/astral-sh/uv
- wandb log in details and be added to the project - https://wandb.ai/site
- You may need to use computa CPU or an alternative if you have an old mac and it's complaining about torch 2.6 version

## To dev
1. To setup: `uv sync` so you will download all needed dependencies
    - To recognise imported files, you may need to run `export PYTHONPATH=./src` in the terminal
    - To run python file: either use `uv run <file name>` will auto use the .venv that uv generates OR you can do `source .venv/bin/activate` to activate your virtual env and then run `python3 <file name>`
    - If you have VSCode on mac, you can do Shift, Command, P to open preferences, and select the .venv. Then you can press 'play' button and that will use it.

## Brainstorming session
![A6098B6E-7365-497E-A0B9-20E573ED6993](https://github.com/user-attachments/assets/7fa56d04-5c6c-4963-b960-1071c5cd63c2)

![8AB8CE0A-D751-48FD-8FD2-8DD4A4176684](https://github.com/user-attachments/assets/02822402-1b06-412a-aed6-67df45cd1a94)

### To Dos
1. Download the hugging face Flickr dataset
2. Link it up with CLIP VIT Encoder
- Process the dataset: need to make it into patches. Also add positional encoding
3. Work out caption -> word2vec -> to vector
4. Concat <start token> + <vectors of patches> + <vectors of caption> + <end token>
5. Write the decoder! And feed in the above
6. Output logits: use the Cross Entropy loss function to train (compare it to the caption that has gone through vec2word)