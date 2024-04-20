import cv2
import socket
import numpy as np
import struct
from model.image2text import Image2Text


HOST = '0.0.0.0'  
PORT = 10000
IMAGE_PATH = "recived_frame.jpg"


def generate_text(_model, _image_path):
    text = _model.run(_image_path)
    print(text)
    return text


def main(_host, _port, _model, _image_path):
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
                        generate_text(_model, _image_path)

if __name__ == "__main__":
    model = Image2Text()
    main(HOST, PORT, model, IMAGE_PATH)