"""
MIT License

Copyright (c) 2019 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import os
import sys
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageDraw, ImageFilter
from crfasrnn import util
from crfasrnn.crfasrnn_model import CrfRnnNet
# Download the model from https://tinyurl.com/crfasrnn-weights-pth
model = CrfRnnNet()
model.eval()
saved_weights_path = "crfasrnn_weights.pth"
model.load_state_dict(torch.load(saved_weights_path))
INPUT_PATH = 'pics/'
OUTPUT_PATH = 'res/'
os.chdir(sys.path[0])


def process_image(input_file):
    # Read the image
    img_data, img_h, img_w, size = util.get_preprocessed_image(input_file)
    original = Image.open(input_file)
    out = model.forward(torch.from_numpy(img_data))
    probs = out.detach().numpy()[0]
    label_im = util.get_label_image(probs, img_h, img_w, size)
    blank = Image.new('RGB', original.size)
    mask = label_im.convert('1').resize(original.size)
    res = Image.composite(original, blank, mask)
    # label_im.save(output_file)
    return res


files = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(INPUT_PATH)] for val in sublist]
# files = [f for f in listdir(INPUT_PATH) if isfile(join(INPUT_PATH, f))]
for file_name in files:
    input_file = file_name
    res = process_image(input_file)
    outpath = os.path.join(OUTPUT_PATH, file_name)
    dirname = os.path.dirname(outpath)
    print(dirname)
    if not os.path.exists(dirname):
    	os.makedirs(dirname)
    res.save(output)

