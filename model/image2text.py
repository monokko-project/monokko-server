"""
https://github.com/mlfoundations/open_clip#fine-tuning-coca
"""

import open_clip
import torch
from PIL import Image


class Image2Text:
  def __init__(self):
    self.model, _, self.transform = open_clip.create_model_and_transforms(
      model_name="coca_ViT-L-14",
      pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )

  def run(self, image_path):
    im = Image.open(image_path).convert("RGB")
    im = self.transform(im).unsqueeze(0)

    with torch.no_grad(), torch.cuda.amp.autocast():
      generated = self.model.generate(im)

    text = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
    
    return text


# m = Image2Text()
# text = m.run("../recived_frame.jpg")
# print(text)