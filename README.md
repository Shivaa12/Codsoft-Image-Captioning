# Codsoft-Image-Captioning
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
import requests
import torch
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

model_raw = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer       = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def show_n_generate(url, greedy = True, model = model_raw):
    image = Image.open(requests.get(url, stream =True).raw)
    encoding = tokenizer.encode_plus(
        "",
        max_length=30,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    pixel_values   = image_processor(image, return_tensors ="pt").pixel_values

    if greedy:
        generated_ids  = model.generate(pixel_values, attention_mask=encoding["attention_mask"], max_new_tokens = 30)
    else:
        generated_ids  = model.generate(
            pixel_values,
            attention_mask=encoding["attention_mask"],
            do_sample=True,
            max_new_tokens = 30,
            top_k=5)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    plt.imshow(np.asarray(image))
    plt.title(generated_text)
    plt.show()

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
show_n_generate(url, greedy = False)
