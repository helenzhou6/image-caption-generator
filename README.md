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
Inference:
![8AB8CE0A-D751-48FD-8FD2-8DD4A4176684](https://github.com/user-attachments/assets/02822402-1b06-412a-aed6-67df45cd1a94)

### To Dos
![A6098B6E-7365-497E-A0B9-20E573ED6993](https://github.com/user-attachments/assets/7fa56d04-5c6c-4963-b960-1071c5cd63c2)

1. Download the hugging face Flickr dataset
2. Link it up with CLIP VIT Encoder (this already has positional encoding embedding and patching - 32pixel x 32pixel = 1 patch by default)
    - Might need to review: 3 colour channels
    - Deal with sizing (photos are different sizes - max of both width and height 470pixel x 500pixel)
3. Extract one (out of 5) caption -> word2vec -> to vector
    - Also handling different lengths 
4. Concat `<start token> + <vectors of patches> + <vectors of caption> + <end token>`
5. Write the decoder! And feed in the above
6. Output logits: use the Cross Entropy loss function to train the base model (compare it to the caption that has gone through vec2word)