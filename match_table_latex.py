import os

os.environ['HF_HOME'] = '/data-pfs/jd/cache/huggingface'

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from utils.qwen2_vl import apply_messages_template, img2str
from utils.io import read_table_infos


def main_routine(processor, model, images):
    # Messages containing multiple images and a text query
    content = []
    for image_path in images:
        content.append({"type": "image", "image": f"file://{image_path}"}, )
    # content.append({""})
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs)  # , max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)


if __name__ == "__main__":
    vl_model = "Qwen/Qwen2-VL-2B-Instruct"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        vl_model, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(vl_model)

    tables_root = 'tables'
    disciplines = os.listdir(tables_root)
    for discipline_i in disciplines:
        discipline_i_root = os.path.join(tables_root, discipline_i)
        paper_ids = os.listdir(discipline_i_root)
        for paper_id in paper_ids:
            paper_id_root = os.path.join(discipline_i_root, paper_id)
            table_image_names = os.listdir(paper_id_root)
