import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig



class  vision2language:
    def __init__(self):
        # bnb_cfg = BitsAndBytesConfig(
        #     # load_in_4bit=False,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.float16,
        #     llm_int8_skip_modules=["mm_projector", "vision_model"],
        # )

        self.model_id = "qresearch/llama-3-vision-alpha-hf"
        self.model = AutoModelForCausalLM.from_pretrained(
            # model_name,
            self.model_id,
            load_in_8bit=False,
            # quantization_config=bnb_cfg,
            trust_remote_code=True,
            # device_map="auto",
            low_cpu_mem_usage=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True
        )


    def run(self, image_path="recieved_frame.png", text="question"):

        image = Image.open(image_path)
    
        output = (
            self.tokenizer.decode(
                self.model.answer_question(image, text, self.tokenizer),
                skip_special_tokens=True,
            )
        )

        return output
