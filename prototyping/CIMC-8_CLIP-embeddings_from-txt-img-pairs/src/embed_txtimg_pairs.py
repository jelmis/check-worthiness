import tqdm
import torch
import numpy as np
from transformers import CLIPTextConfig, CLIPTokenizerFast, CLIPProcessor, CLIPModel


from tqdm.auto import tqdm


def embed_split(texts, images, batch_size = 16):

    image_arr = None
    text_arr = None

    device = "cuda" if torch.cuda.is_available() else \
        ("mps" if torch.backends.mps.is_available() else "cpu")
    
    model_id = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)

    for i in tqdm(range(0, len(images), batch_size)):

        # Tokenize and embed the batch texts
        batch_txts = texts[i:i+batch_size]
        inputs = tokenizer(batch_txts, padding=True, return_tensors="pt").to(device)
        batch_txt_emb = model.get_text_features(**inputs)

        # Process and embed the batch images
        batch = images[i:i+batch_size]
        batch = processor(
            text=None,
            images=batch,
            return_tensors='pt',
            padding=True
        )['pixel_values'].to(device)

        batch_emb = model.get_image_features(pixel_values=batch)
        batch_emb = batch_emb.squeeze(0)
        batch_emb = batch_emb.cpu().detach().numpy()

        if image_arr is None:
            image_arr = batch_emb
            text_arr = batch_txt_emb
        else:
            image_arr = np.concatenate((image_arr, batch_emb), axis=0)
            text_arr = np.concatenate((text_arr,batch_txt_emb), axis = 0)

    return text_arr, image_arr



