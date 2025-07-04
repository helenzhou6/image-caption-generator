# Image Caption Generator
Aim: To take an image and generate a caption with **magic** (a.k.a a CLIP Visual Encoder + Custom Decoder)
MLX week 4.

Screenshot:
![Image of inference working](https://github.com/user-attachments/assets/ffc784f0-d6f2-4285-a923-1664ea2bc24d)

## Pre-requisities 
- Python 3.10
- uv - install via https://github.com/astral-sh/uv
- wandb log in details and be added to the project - https://wandb.ai/site or in remote terminal 
- You may need to use computa CPU or an alternative if you have an old mac and it's complaining about torch 2.6 version

### Remote instructions
1. Get your GPU and connect to it (either via VSCode SSH plugin or within the terminal)
2. Clone the repo:  https://github.com/helenzhou6/image-caption-generator.git
3. Export env variables needed `export WANDB_API_KEY="your_wandb_key" GITHUB_NAME="your_github_name" GITHUB_EMAIL="your_email@example.com"`
4. To clone and install eveything, run `chmod +x ./gpu-script.sh` and `./gpu-script.sh`
5. Run any commands in `tmux` (will have a green at the bottom) - i.e. run `tmux`! This will still run commands in the background even if you get kicked out (SSH connection fails)
    - To see all tmux sessions, run `tmux ls` and then run `tmux a -t <session name/number>` to go back to that session

## To dev
1. To setup: `uv sync` so you will download all needed dependencies
    - To recognise imported files, you may need to run `export PYTHONPATH=./src` in the terminal
    - To run python file: either use `uv run <file name>` will auto use the .venv that uv generates OR you can do `source .venv/bin/activate` to activate your virtual env and then run `python3 <file name>`
    - If you have VSCode on mac, you can do Shift, Command, P to open preferences, and select the .venv. Then you can press 'play' button and that will use it.
2. (Optional): run `uv run src/model/dataset.py` that will create the training and validation dataset as pickle files & upload to wandb. This will not need to be run if anyone has run this before and artifacts are already in wandb.
3. Run `uv run src/model/train.py` to train and evaluate the model (against METEOR)
4. To run inference - run `uv run streamlit run src/model/inference.py` that will spin up the website, and generate captions based on images being uploaded

## Brainstorming session
Inference:
![8AB8CE0A-D751-48FD-8FD2-8DD4A4176684](https://github.com/user-attachments/assets/02822402-1b06-412a-aed6-67df45cd1a94)

### To Dos
![A6098B6E-7365-497E-A0B9-20E573ED6993](https://github.com/user-attachments/assets/7fa56d04-5c6c-4963-b960-1071c5cd63c2)

1. Download the hugging face Flickr dataset ✅
2. Link it up with CLIP ViT Encoder (this already has positional encoding embedding and patching - 32pixel x 32pixel = 1 patch by default) ✅
    - Might need to review: 3 colour channels ✅
    - Deal with sizing (photos are different sizes - max of both width and height 470pixel x 500pixel) ✅
    - Understand the shape of patch_embeddings patch tensor ✅
    - Change encoder so it takes in the whole dataset (create dataloader) ✅ 
    - Need to add padding to the caption embedding, to the max length in the batch ✅ 
    - Add positional embedding ✅ 
    - Project image to same 512 dim as text ✅ 
    - Concat (start token + visual embedding + text embedding + end token etc) ✅ 
3. Extract one (out of 5) caption -> word2vec -> to vector ✅
    - Also handling different lengths (train_datset max length 402 - test_datasest 342)
4. Concat `<start token> + <vectors of patches> + <vectors of caption> + <end token>` ✅
5. Write the decoder! ✅
    - Feed in the above ✅
    - Add decoder layers ✅
    - Add in mask to self attention ✅
6. Output logits: use the Cross Entropy loss function to train the base model (compare it to the caption that has gone through vec2word) ✅
7. Evaluation - using validation dataset ✅
    - Nice to have: during training, will output some text! ✅
8. Sweeps
9. Inference ✅


Conceptual Questions
1. Are the image encoder & text encoders pre-trained ?
_YES_
   
2. Are they models (i.e .pth files like we would get after training a model) or layers (i.e classes that we would import/use from torch library? 
_Models - we are loading the .pth, which is inside the CLIP transformer from the transformers library_

4. What are they trained on?
_400 million (image,text) pairs collected from public data by OpenAI, trained using contrastive loss function. _
   
6. When we use them, they turn images or captions into vectors of specific dimensions (eg. 768 etc). Does this mean that "meaning" is then 'baked into' the output from the model? eg. the outputted vector contains only contextual meaning about what the image represents

_YES. The model maps the information from each image (patch) into a vector space of size embed_dim representing 'features' of the patches (e.g., 'redness', 'edgeness' in human terms). This mapping is a (lower) resolution equivalent to the original image (patch) attributes that a neural network can 'understand'. It contains the 'meaning' of the image (patches) and allows the model to assess similarity via a dot product (we humans can do this just by looking). _

8. So what is the Decoder learning? What are the weights and biases really representing? i.e what is it learning, and what do we backpropogate it to train it to be more accurate.  

### Bonus 
1. Try pooling 5 captions for embeddings and test against single embedded caption
2. Create image-caption pairs for all 5 captions for a given image & re-train
3. Handle non square images, by including rectangle patches? 
4. Use Qwen text encoder/embedding 

# Task 2 -  QWEN encoder

Aim: Have a custom generated dataset, in our case a nutritional label dataset. Using pre-trained Qwen model, with system prompts that will have the model pretend to be a grandma that is working for a confectionary company and trying to persuade you to eat sugary foods.
- Side note: luckily we weren't able to turn the model 'bad', it always recommended not having high sugar content!

## Brainstorming
![Image of brainstorming whiteboarding](https://github.com/user-attachments/assets/b64590b9-e036-49e2-b717-eebbdddfb7f0)

## To run
### Custom dataset generator
1. To run `export HF_API_TOKEN=<API KEY>`  and `uv run src/custom-dataset-generator/generate_captions.py` that will take all the images in nutrition labels, and then generate captions for it and save to huggingface
- Link to dataset: https://huggingface.co/datasets/sugarbot/nutrition-labels-dataset/viewer/default/train?row=2&views%5B%5D=train

## To run the model
1. Run `uv run src/qwen/train.py`
2. To run the inference: `uv run streamlit run src/qwen/inference.py`