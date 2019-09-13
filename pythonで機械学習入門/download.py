import numpy as np
import matplotlib.pyplot as plt

import chainer.optimizers as Opt
import chainer.functions as F
import chainer.links as L
import chainer
import chainer.serializers as ser

from chainer import Variable,Chain,config,cuda
from tqdm import tqdm

import os
import PIL.Image as im        

import urllib.error
import urllib.request

def download_image(url, dst_path):
    try:
        data = urllib.request.urlopen(url).read()
        with open(dst_path, mode="wb") as f:
            f.write(data)
    except urllib.error.URLError as error:
        print(error)
        
def download_figs():
    
    url_list = []
    for i in range(19):
        url = "https://github.com/mohzeki222/ohm_princess/blob/master/notes/princess_fig/kisaki{:02}.jpg".format(i)
        url_list.append(url)
    download_dir = "princess_fig"
    for url in url_list:
        filename = os.path.basename(url)
        dst_path = os.path.join(download_dir, filename)
        download_image(url, dst_path)
        
    url_list = []
    for i in range(8):
        url = "https://github.com/mohzeki222/ohm_princess/blob/master/notes/white_fig/sira{:02}.jpg".format(i)
        url_list.append(url)
    download_dir = "white_fig"
    for url in url_list:
        filename = os.path.basename(url)
        dst_path = os.path.join(download_dir, filename)
        download_image(url, dst_path)

download_figs()
