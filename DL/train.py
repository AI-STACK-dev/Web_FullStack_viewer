import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import skimage.transform
import scipy.ndimage as nd
import scipy.ndimage.morphology as mp
import random
import time

from typing import Any, Callable, cast, Dict, List, Optional, Tuple

root_dir = '/mnt/hsyoo/WSI_image/' #수정
batch_size = 100
learning_rate = 3e-4
num_epochs = 50
num_workers = 4
'''
SE size는 64 x 64
'''
SE = (64, 64) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def get_diff(original, annotation):
    b_a = annotation - original
    
    masks = b_a != 0
    b_a[masks] = 255
    
    h,w = b_a.shape[:2]
    masks = np.zeros([h+2, w+2], np.uint8)
    cv.floodFill(b_a, masks, (30,30), (255, 255, 255))
    
    mask = b_a[:,:,0].copy()
    mask[mask == 0] = 1
    mask[mask == 255] = 0
    
    return mask

def get_normal(original, annotation):
    mask = original[:,:,0].copy()
    mask[mask == 255] = 0
    mask[mask != 0] = 1
    
    mask = mask - get_diff(original, annotation)
    mask[mask == 255] = 1
    return mask

class Dataset_train(Dataset):
    def read_data_set(self):
        folder_names = ['(0)_NOLN_metastasis', '(1)_LN_metastasis']
        label_names = ['(0)_normal', '(1)_cancer']
        all_img_files = []
        all_labels = []
        samples = []
        
        
        for folder_name in folder_names:
            for slide in sorted(os.listdir(os.path.join(self.data_set_path, folder_name, 'original'))):
                '''
                validation으로 두는 두 개는 Trainset에서는 제외한다.
                '''
                # if slide.split('.')[0].split('_')[-1] == format(self.val[0], '03'):
                #     continue
                # if slide.split('.')[0].split('_')[-1] == format(self.val[1], '03'):
                #     continue
                # if slide.split('.')[0].split('_')[-1] == format(self.val[2], '03'):
                #     continue
                if slide.split('.')[0].split('_')[-1] in self.val:
                    continue
                '''
                확장자가 png가 아닌 파일은 제외한다.
                '''
                if slide.split('.')[-1] != 'png':
                    continue
                
                print("Check slide input : ", slide)
                annotation = np.array(Image.open(os.path.join(self.data_set_path, folder_name, 'label', slide)))
                original = np.array(Image.open(os.path.join(self.data_set_path, folder_name, 'original', slide)))
                
                for label_name in label_names:
                    if label_name == '(0)_normal':
                        b_a = get_normal(original, annotation)
                    else:
                        b_a = get_diff(original, annotation)
                        
                    erosion = mp.binary_erosion(b_a, np.ones(SE))
                    cat = np.concatenate((original, erosion[:,:,np.newaxis]), axis = 2)
                    idx = np.arange(len(np.where(erosion)[0]))
                    np.random.shuffle(idx)
                    x = np.where(erosion)[0][idx]
                    y = np.where(erosion)[1][idx]
                    cnt = 0
                    j = 0
                    
                    while cnt < 1000:
                        flag = 0
                        while flag != 1:
                            '''
                            Augmentation으로 RandomCrop을 이용할 예정
                            '''
                            temp = cat[x[j] - 160 : x[j] +160, y[j] - 160 : y[j]+160 ,:3]
                            if temp.shape[0] == 320 and temp.shape[1] == 320:
                                flag = 1
                                temp = Image.fromarray(temp).convert("RGB")
                                j += 1
                            else:
                                j += 1
                        cnt += 1
                        all_img_files.append(temp)
                        
                        if label_name == '(0)_normal':
                            label = 0
                        else:
                            label = 1
                            
                        all_labels.append(label)
                        samples.append((temp, label))
                        
        return all_img_files, all_labels, len(all_img_files), len(label_names), samples
                        
                
    
    def _find_classes(self, dir : str) -> Tuple[List[str], Dict[str, int]]:
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name : i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def __init__(self, data_set_path, val, transforms = None):
        self.data_set_path = data_set_path
        classes, class_to_idx = self._find_classes(self.data_set_path)
        self.val = val
        self.image_files_path, self.labels, self.length, self.num_classes, samples = self.read_data_set()
        self.imgs = samples
        self.transforms = transforms
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]
        
    
    def __getitem__(self, index):
        image = self.image_files_path[index]
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return {'image' : image, 'label' : self.labels[index]}
    
    def __len__(self):
        return self.length

class Dataset_valid(Dataset):
    def read_data_set(self):
        folder_names = ['(0)_NOLN_metastasis', '(1)_LN_metastasis']
        label_names = ['(0)_normal', '(1)_cancer']
        all_img_files = []
        all_labels = []
        samples = []
        
        
        for folder_name in folder_names:
            for slide in sorted(os.listdir(os.path.join(self.data_set_path, folder_name, 'original'))):
                if slide.split('.')[-1] != 'png':
                    continue
                if slide.split('.')[0].split('_')[-1] in self.val:
                    
                    annotation = np.array(Image.open(os.path.join(self.data_set_path, folder_name, 'label', slide)))
                    original = np.array(Image.open(os.path.join(self.data_set_path, folder_name, 'original', slide)))

                    for label_name in label_names:
                        if label_name == '(0)_normal':
                            b_a = get_normal(original, annotation)
                        else:
                            b_a = get_diff(original, annotation)
                            
                        erosion = mp.binary_erosion(b_a, np.ones(SE))
                        cat = np.concatenate((original, erosion[:,:,np.newaxis]), axis = 2)
                        idx = np.arange(len(np.where(erosion)[0]))
                        np.random.shuffle(idx)
                        x = np.where(erosion)[0][idx]
                        y = np.where(erosion)[1][idx]
                        cnt = 0
                        j = 0
                        
                        while cnt < 1000:
                            flag = 0
                            while flag != 1:
                                temp = cat[x[j] - 160 : x[j] +160, y[j] - 160 : y[j]+160 ,:3]
                                if temp.shape[0] == 320 and temp.shape[1] == 320:
                                    flag = 1
                                    temp = Image.fromarray(temp).convert("RGB")
                                    j += 1
                                else:
                                    j += 1
                            cnt += 1
                            all_img_files.append(temp)
                            if label_name == '(0)_normal':
                                label = 0
                            else:
                                label = 1
                            all_labels.append(label)
                            samples.append((temp, label))
                        
        return all_img_files, all_labels, len(all_img_files), len(label_names), samples
                        
                
    
    def _find_classes(self, dir : str) -> Tuple[List[str], Dict[str, int]]:
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name : i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def __init__(self, data_set_path, val, transforms = None):
        self.data_set_path = data_set_path
        classes, class_to_idx = self._find_classes(self.data_set_path)
        self.val = val
        self.image_files_path, self.labels, self.length, self.num_classes, samples = self.read_data_set()
        self.imgs = samples
        self.transforms = transforms
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]
        
    
    def __getitem__(self, index):
        image = self.image_files_path[index]
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return {'image' : image, 'label' : self.labels[index]}
    
    def __len__(self):
        return self.length
def train(k):
    seed = 42
    random_seed(seed, True)
    

    tra = [
    transforms.RandomHorizontalFlip(), 
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.RandomAffine((-10,10), shear=10, scale=(0.9, 1.2)),
    transforms.RandomCrop(224),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    val = [
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    '''
    1~5까지 도는데, 1번의 경우 validation으로 NOLN, LN 1,2번 슬라이드를 사용한다!
    # of number 
    Trainset : 3,4,5,6,7,8,9,10(x2)
    Validset : 1,2(x2)
    '''
    num = 29//5
    # for k in range(1,6):
    valnum_list = []
    for i in range(num-1,-1,-1):
        valnum_list.append(format(num*k-i, '03'))
    
    print(f'validation number : {valnum_list}')
    end = time.time()
    trainset = Dataset_train(data_set_path = root_dir, val = valnum_list, transforms = transforms.Compose(tra))
    validset = Dataset_valid(data_set_path = root_dir, val = valnum_list, transforms = transforms.Compose(val))
    
    print('Load dataset: {0:.2f} s'.format(time.time() - end))
    
    print('trainset size : ', len(trainset))
    print('validset size : ', len(validset))
    train_iter = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle = True, num_workers = num_workers)#, pin_memory = True
    valid_iter = torch.utils.data.DataLoader(validset,batch_size=batch_size, shuffle=False, num_workers = num_workers)
    
    net = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained = True)
    net.fc = nn.Linear(2048, 2)
    
    torch.nn.init.xavier_uniform_(net.fc.weight) #fine-tunning
    net = net.to(device)

    loss = torch.nn.CrossEntropyLoss() # loss
    alg = torch.optim.Adam(net.parameters(),lr=learning_rate)
    
    loss_train = np.array([])
    loss_valid = np.array([])
    accs_train = np.array([])
    accs_valid = np.array([])
    # num1_tmp = 3*k-2
    # num2_tmp = 3*k-1
    # num3_tmp = 3*k
    # num1 = format(num1_tmp, '03')
    # num2 = format(num2_tmp, '03')
    # num3 = format(num3_tmp, '03')

    for epoch in range(num_epochs):
        print("epoch : ", epoch)
        end = time.time()
        net.train()
        i=0
        l_epoch = 0
        correct = 0
        l_epoch_val = 0
        for i_batch, item in enumerate(train_iter):
            #print("Train - i_batch : ", i_batch)
            i=i+1
            X,y = item['image'].to(device), item['label'].to(device)
            y_hat = net(X)
            y_hat= F.softmax(y_hat, dim = 1)
            l=loss(y_hat,y)
            correct += (y_hat.argmax(dim=1)==y).sum()
            l_epoch+=l
            alg.zero_grad()
            l.backward()
            alg.step()
        loss_train = np.append(loss_train,l_epoch.cpu().detach().numpy()/i)
        accs_train = np.append(accs_train,correct.cpu()/np.float(len(trainset)))

        correct = 0
        i = 0
        net.eval()
        with torch.no_grad():
            for i_batch, item in enumerate(valid_iter):
                #print("Valid - i_batch : ", i_batch)
                i +=1
                X,y = item['image'].to(device), item['label'].to(device)
                y_hat=net(X)
                y_hat= F.softmax(y_hat, dim = 1)
                l = loss(y_hat, y)
                correct += (y_hat.argmax(dim=1)==y).sum()
                l_epoch_val += l
        accs_valid = np.append(accs_valid,correct.cpu()/np.float(len(validset)))
        loss_valid = np.append(loss_valid, l_epoch_val.cpu().detach().numpy()/i)

        fig = plt.figure(figsize = (12, 6))
        ax = fig.add_subplot(1,2,1)
        plt.plot(loss_train,label='train loss')
        plt.plot(loss_valid, label='valid loss')
        plt.legend(loc='lower left')
        plt.title('epoch: %d '%(epoch+1))

        ax = fig.add_subplot(1,2,2)
        plt.plot(accs_train,label='train accuracy')
        plt.plot(accs_valid,label='valid accuracy')
        plt.legend(loc='lower left')
        plt.pause(.0001)
        plt.show()
        fig.savefig(f'/mnt/hsyoo/loss/loss_val_{valnum_list[0]}_{valnum_list[1]}_{valnum_list[2]}_{valnum_list[3]}_{valnum_list[4]}.png')
        fig.savefig(f'/mnt/pathology/hsyoo/loss/loss_val_{valnum_list[0]}_{valnum_list[1]}_{valnum_list[2]}_{valnum_list[3]}_{valnum_list[4]}.png')

        print('train loss: ',loss_train[-1])
        print('valid loss: ', loss_valid[-1])
        print('train accuracy: ',accs_train[-1])
        print('valid accuracy: ',accs_valid[-1])
        print('1 epoch : {0:.2f} s'.format(time.time() - end))
            
        print("epoch success")
        torch.save(net.state_dict(),f'/mnt/hsyoo/n_pth_58/step1_val_{valnum_list[0]}_{valnum_list[1]}_{valnum_list[2]}_{valnum_list[3]}_{valnum_list[4]}_e_{epoch}.pth') #수정
        torch.save(net.state_dict(),f'/mnt/pathology/hsyoo/n_pth_58/step1_val_{valnum_list[0]}_{valnum_list[1]}_{valnum_list[2]}_{valnum_list[3]}_{valnum_list[4]}_e_{epoch}.pth')
        print("pth file save success")
    
if __name__ == '__main__':
    for k in range(3,6):
        train(k)
