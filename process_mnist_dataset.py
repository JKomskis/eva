import numpy as np
import gzip
import struct
from PIL import Image
import os
import cv2
import glob

image_size = 28

# Based on code from https://gist.github.com/xkumiyu/c93222f2dce615f4b264a9e71f6d49e0
def create_img_files(input_path, image_path):
    images = None
    with gzip.open(input_path) as file:
            file.read(4)
            N, = struct.unpack('>i', file.read(4))
            file.read(8)

            images = np.empty((N, 784), dtype=np.uint8)

            for i in range(N):
                for j in range(784):
                    images[i, j] = ord(file.read(1))

    os.makedirs(image_path, exist_ok=True)
    for (i, image) in enumerate(images):
        filepath = f'{image_path}/{i}.jpg'
        Image.fromarray(image.reshape(image_size, image_size)).save(filepath)

def create_video_from_images(image_path, output_file):
    writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'avc1'), 30, (image_size, image_size))
    for filename in glob.glob(f'{image_path}/*'):
        img = cv2.imread(filename)
        for i in range(0, 6):
            writer.write(img)

if __name__ == "__main__":
    input_path = 'data/MNIST/raw/train-images-idx3-ubyte.gz'
    image_path = 'data/MNIST/processed/train'
    create_img_files(input_path, image_path)
    create_video_from_images(image_path, 'data/MNIST/train_long.mp4')
