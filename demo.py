import cv2
import socket
import numpy as np
import struct

# from model.vision2language import vision2language
from model.v2l_claude import vision2language

HOST = '0.0.0.0'  
PORT = 10000
IMAGE_PATH = "recieved_frame.png"

def main(_host, _port, _image_model, _nlp_model, _image_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # launch and wait
        s.bind((_host, _port))
        s.listen(1)

        while True:
            print('Waiting for connection...')

            conn, addr = s.accept()
            with conn:
                print('- Connected by', addr)

                conn_type = conn.recv(1024)
                # print("- device type :", conn_type)
                if  conn_type == b"camera":
                    print("--> recived image data...")
                    while True:

                        # size of frame data
                        data = conn.recv(4)
                        if not data:
                            break
                        size = struct.unpack('!I', data)[0]
                        
                        # recieve frame data
                        data = b''
                        while len(data) < size:
                            packet = conn.recv(size - len(data))
                            
                            if not packet:
                                break
                            data += packet

                        # decode
                        frame_data = np.frombuffer(data, dtype=np.uint8)
                        frame = cv2.imdecode(frame_data, 3)

                        cv2.imwrite( _image_path, frame)
                        # text = generate_text(_model, _image_path)
                        text = _image_model.run("recieved_frame.png", "この画像について説明してください。”モノ”について注意深く、可能な限り詳しく説明してください。")
                        # text = "Hello World"
                        conn.sendall(text.encode("utf-8"))
                        # result = _nlp_model.trans_en_to_jp(text)
                        result = "こんにちは、世界"
                        conn.sendall(result.encode("utf-8"))


if __name__ == "__main__":
    # model = Image2Text()
    model = vision2language()
    # nlp_en = NLPmodel("en_core_web_sm")
    main(HOST, PORT, model, None, IMAGE_PATH)