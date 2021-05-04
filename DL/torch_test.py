import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

import os
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2
import sys
import skimage.transform
import base64


# base64 받음
inputs = sys.stdin.read()
inputs = str(inputs.rstrip('\n'))

#binary_arry = base64.b64decode(inputs)
#binary_np = np.frombuffer(binary_arry, dtype=np.uint8)

# data cv2 np convert
#img_np = cv2.imdecode(binary_np, cv2.IMREAD_ANYCOLOR)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def get_mask(original, annotation):
    
    b_a = annotation - original
    
    masks = b_a != 0
    b_a[masks] = 255
    
    h,w = b_a.shape[:2]
    masks = np.zeros([h+2, w+2], np.uint8)
    cv.floodFill(b_a, masks, (30,30), (255, 255, 255))
    
    mask = b_a[:,:,0].copy()
    mask[mask == 0] = 1
    mask[mask == 255] = 0
    
    new_mask = original[:,:,0] != 255
    #new_mask[mask == 1] = 2
    new_mask[new_mask == True] = 1
    new_mask[new_mask == False] = 0
    new_mask[mask == 1] = 2
    
    return mask, new_mask

def max_count(stride, original):
    n = 0
    while (256+stride *n) <= original.shape[0]:
        n +=1
    max_i = n-1
    
    n=0
    while(256 + stride*n) <= original.shape[1]:
        n +=1
    max_j = n-1
    
    return max_i, max_j

def get_probmap(img_array, stride, max_i, max_j):
  overlay = np.zeros(shape = (max_i, max_j))
  for i in range(max_i):
    for j in range(max_j):
      temp = img_array[(stride * i) : 256 + (stride * i), (stride*j) : 256 + (stride*j), :]
      temp = Image.fromarray(temp)
      temp = transform(temp)
      temp = temp.unsqueeze(0)
      output = resnet_step1(temp.to(device))
      output = torch.nn.functional.softmax(output[0], dim = 0)
      out_cancer = output[1]
      overlay[i][j] = out_cancer
  return overlay

stride = 128
seed = 42
random_seed(seed,True)
# 
# 여기서 어떤 형태의 이미지를 받는다. 그거를 Image.open을 통해서 받을 예정(json file)
#input_slide = Image.open("/home/hufsbme/Desktop/capstoneWebviewer/server_work_download_upload_neww/server_work_download_upload/public/img/LN_meta_image_004.png")
#inputs = sys.stdin.read()
#binary_arr = base64.b64decode(inputs)
#input_slide = np.frombuffer(binary_arr, dtype=np.uint8)
#print(binary_np.shape)

#input = inputs.split('\')[0]
#inputs = "NOLN_meta_image_001.png"
name = inputs.split('_')[0]
if name == 'NOLN':
    folder_name = '(0)_NOLN_metastasis'
else:
    folder_name = '(1)_LN_metastasis'
# ㅇㅕ기서 inputs name을 받는닫.
input_slide = np.array(Image.open(f"/mnt/hsyoo/WSI_image/{folder_name}/original/{inputs}"))

transform = torchvision.transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]) 

resnet_step1 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained = True)
resnet_step1.fc = nn.Linear(2048, 2)
for filename in os.listdir(os.path.join('/mnt/hsyoo/n_pth_58/')):
    if inputs.split('.')[0].split('_')[-1] in filename:
        resnet_step1.load_state_dict(torch.load(f'/mnt/hsyoo/n_pth_58/{filename}'))
        break

#resnet_step1.load_state_dict(torch.load('/mnt/pathology/pth_20/step1_val_005_006.pth'))
resnet_step1 = resnet_step1.to(device)
resnet_step1.eval()

max_i, max_j = max_count(stride = stride, original = input_slide)
stime = time.time()
overlay = get_probmap(img_array = input_slide, stride = stride, max_i = max_i, max_j = max_j)
resized_overlay = skimage.transform.resize(overlay, input_slide.shape[:2])
#print("overlay complete")
fig = plt.figure(dpi = 200)
plt.imshow(input_slide)
plt.imshow(resized_overlay, cmap = 'jet', alpha = 0.5)
plt.axis('off')
#plt.title(f'{inputs}')

fig.savefig('/mnt/hjjang/project/public/img/result.png')

#time.sleep(15)


def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

img_np = figure_to_array(fig)

_, imen = cv2.imencode('.png', img_np)
imenb = imen.tobytes()

 # python의 endcode는 base64 문자열의 bytes타입으로 바꿔주므로, 다시 문자열로 decode
result = base64.b64encode(imenb).decode()
print(result)

#print(overlay)
#print("Overlay Done :", overlay.shape)
#print("time : ", time.time() - stime)



