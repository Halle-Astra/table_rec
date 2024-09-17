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



if __name__ == "__main__":
    vl_model = "Qwen/Qwen2-VL-2B-Instruct"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        vl_model, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(vl_model)

    tables_root = 'tables'
    disciplines = os.listdir(tables_root)



