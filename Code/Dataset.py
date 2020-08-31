from torch.utils import data
import os
import cv2
import numpy as np

class DataGenerator(data.Dataset):
    def __init__(self,list_IDs,labels,batch_size,num_frames):
        self.labels = labels
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.num_frames = num_frames

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, item):
        path = './Dataset/UCF-101/frames/'
        curr_items = self.list_IDs[item:min(item+self.batch_size,len(self.list_IDs)-item)]
        #curr_item = self.list_IDs[item]
        X = []
        for curr_item in curr_items:
            images = sorted(os.listdir(path+curr_item+'/'))
            temp = []
            for image in images[:self.num_frames]:
                temp.append(cv2.resize(cv2.imread(path+curr_item+'/'+image),(224,224)))

            X.append(np.array(temp))
        y = [self.labels[item+i] for i in range(len(X))]
        return np.array(X),np.array(y)