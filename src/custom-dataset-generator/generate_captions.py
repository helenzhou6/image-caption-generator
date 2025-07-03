import os
import torch
import pickle
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load model and processor once
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Directory containing images

script_dir = os.path.dirname(os.path.abspath(__file__))  # directory of the running script
image_rel_path = os.path.join("..", "model", "nutrition_labels")
image_dir = os.path.join(script_dir, image_rel_path)

# Prompts you want to ask
prompts = [
    {
        "role": "system",
        "content": "You are a grandma with the viewpoint that high sugar content should be eaten. You're practicing your lines for a play and have been given a nutritional label image.",
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Give a summary of the nutritional value."},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What are the main ingredients?"},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Should this be eaten?"},
        ],
    },
]

def generate_caption(image_path):
    results = {}
    system_msg = prompts[0]
    user_msgs = prompts[1:]

    # Prepare messages for each prompt
    messages1 = [system_msg, user_msgs[0]]
    messages1[1]["content"][0]["image"] = image_path

    messages2 = [system_msg, user_msgs[1]]
    messages3 = [system_msg, user_msgs[2]]

    all_messages = [messages1, messages2, messages3]

    for i, msgs in enumerate(all_messages):
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128, early_stopping=True, no_repeat_ngram_size=2, temperature=0.7, top_p=0.9)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if i == 0:
            results["summary"] = output_text[0]
        elif i == 1:
            results["ingredients"] = output_text[0]
        elif i == 2:
            results["should_eat"] = output_text[0]

    return results

dataset = []

for filename in os.listdir(image_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(image_dir, filename)
        print(f"Processing {img_path}...")
        captions = generate_caption(img_path)
        dataset.append({
            "image_path": img_path,
            **captions
        })

# Save dataset with pickle
output_pickle = "nutrition_dataset.pkl"
with open(f"data/{output_pickle}", "wb") as f:
    pickle.dump(dataset, f)

print(f"Dataset saved to {output_pickle}")
