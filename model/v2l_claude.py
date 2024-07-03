import requests
from PIL import Image
import io
import base64
from model.env import CLAUDE_API_KEY

class vision2language:
    def __init__(self):
        self.api_key = CLAUDE_API_KEY
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

    def run(self, image_path="received_frame.png", text="質問"):
        # 画像を開いてbase64エンコードする
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # APIリクエストの準備
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
        }

        # APIリクエストを送信
        response = requests.post(self.api_url, headers=self.headers, json=payload)

        if response.status_code == 200:
            return response.json()['content'][0]['text']
        else:
            return f"Error: {response.status_code}, {response.text}"
        


if __name__ == "__main__":
    v2l = vision2language()
    print(v2l.run("recieved_frame.png", "この画像について説明してください。”モノ”について注意深く、可能な限り詳しく説明してください。"))