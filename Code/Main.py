from Model import BNInception_gsm
import pandas as pd
import numpy as np
import torch
from torch import nn
import os
import random
import cv2
from Dataset import DataGenerator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
from torch.backends import cudnn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from CosineAnnealingLR import WarmupCosineLR
import pickle
import datetime

num_classes = 101

pretrained_settings = {
    'bninception': {
        'imagenet': {
            # Was ported using python2 (may trigger warning)
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth',
            # 'url': 'http://yjxiong.me/others/bn_inception-9f5701afb96c8044.pth',
            'input_space': 'BGR',
            'input_size': [3, 224, 224],
            'input_range': [0, 255],
            'mean': [104, 117, 128],
            'std': [1, 1, 1],
            'num_classes': 1000
        }
    }
}

def bninception(num_classes=1000, pretrained='imagenet'):
    r"""BNInception model architecture from <https://arxiv.org/pdf/1502.03167.pdf>`_ paper.
  """
    model = BNInception_gsm(num_classes=101)
    if pretrained is not None:
        settings = pretrained_settings['bninception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        states = model_zoo.load_url(settings['url'])
        del states['last_linear.weight']
        del states['last_linear.bias']

        model.load_state_dict(states, strict=False)
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    return model

def evaluate(preds,labels):
    correct = 0
    correct_class = dict([(x,0) for x in range(101)])
    for i in range(len(preds)):
        if preds[i]==labels[i]:
            correct += 1
            correct_class[preds[i]] += 1
            print(i)
            print(labels[i])

    return correct/len(preds),correct_class

def predict(model,num_frames,class_dict):
    video_file = 'Archery/v_Archery_g07_c03'
    path = './Dataset/UCF-101/frames/'
    video_frames = sorted(os.listdir(path + video_file + '/'))
    images = []
    for image in video_frames[:num_frames]:
        images.append(cv2.resize(cv2.imread(path + video_file + '/' + image), (224, 224)))

    images = np.array(images)
    model.eval()
    output = torch.mean(
        model(torch.from_numpy(images.reshape((num_frames, 3, 224, 224))).float().cuda()),
        dim=[0]).view(1, -1)

    prediction = np.argmax(output.cpu().data.numpy())
    print('Video file name: ',video_file)
    print('Model prediction: ',class_dict[prediction])

def main():
    classes_dict = {}
    path = './Dataset/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/'
    with open(path + 'classInd.txt', 'r') as f:
        classes_dict = dict([(int(x.split()[0]) - 1,x.split()[1]) for x in f.read().strip().split('\n')])

    partition = {'training': [], 'validation': [], 'testing': []}
    with open(path + 'trainlist03.txt', 'r') as f:
        data = [x.split() for x in f.read().strip().split('\n')]

    labels = {'training': [], 'validation': [], 'testing': []}
    random.seed(1)

    classes = dict([(x,[]) for x in range(num_classes)])
    for i in range(len(data)):
        classes[int(data[i][1])-1].append(i)

    training_sample = []
    validation_sample = []
    for i in classes.keys():
        training_sample.extend(random.sample(classes[i],20))
        classes[i] = [x for x in classes[i] if x not in training_sample]
        validation_sample.extend(random.sample(classes[i], 5))

    training_sample = set(training_sample)
    validation_sample = set(validation_sample)
    X_train = [data[x][0][:-4] for x in range(len(data)) if x in training_sample]
    y_train = [int(data[x][1]) - 1 for x in range(len(data)) if x in training_sample]
    X_val = [data[x][0][:-4] for x in range(len(data)) if x in validation_sample]
    y_val = [int(data[x][1]) - 1 for x in range(len(data)) if x in validation_sample]

    for i, j in enumerate(X_train):
        partition['training'].append(j)
        labels['training'].append(y_train[i])

    for i, j in enumerate(X_val):
        partition['validation'].append(j)
        labels['validation'].append(y_val[i])

    num_frames = 30
    batch_size = 3

    training_set = DataGenerator(partition['training'], labels['training'], batch_size=batch_size,
                                 num_frames=num_frames)
    validation_set = DataGenerator(partition['validation'], labels['validation'], batch_size=batch_size,
                                   num_frames=num_frames)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.cuda.empty_cache()
    cudnn.benchmark = True
    print('Device: ', device)

    model = bninception().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
    num_epochs = 100

    scheduler = WarmupCosineLR(optimizer=optimizer,milestones=[10,num_epochs],warmup_iters=10,min_ratio=1e-7)

    model.train(True)
    print('Started training')
    losses = []
    for epoch in range(num_epochs):
        starttime = datetime.datetime.now()
        idx = 0
        running_loss = 0
        num_batches = 0
        while (idx < len(training_set)):
            curr_batch, curr_labels = training_set[idx]
            if len(curr_batch)==0:
                break

            outputs = torch.mean(
                model(torch.from_numpy(curr_batch[0].reshape((curr_batch.shape[1], 3, 224, 224))).float().cuda()),
                dim=[0]).view(1, -1)
            for i in range(1, len(curr_batch)):
                output = model(
                    torch.from_numpy(curr_batch[i].reshape((curr_batch.shape[1], 3, 224, 224))).float().cuda())
                outputs = torch.cat((outputs, torch.mean(output, dim=[0]).view(1, -1)), dim=0)

            loss = criterion(outputs, torch.from_numpy(curr_labels).cuda())
            losses.append(loss.item())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            idx += len(curr_batch)
            num_batches += 1

        endtime = datetime.datetime.now()
        print('Average loss for epoch {} is: {}'.format(epoch,running_loss/num_batches))
        print('Execution time: ',endtime-starttime)

    torch.save(model.state_dict(),'./Model.pt')

    model.eval()
    
    idx = 0
    predictions = []
    while (idx < len(labels['validation'])):
        curr_batch, curr_labels = validation_set[idx]
        if len(curr_labels)==0:
            break
        outputs = torch.mean(
            model(torch.from_numpy(curr_batch[0].reshape((curr_batch.shape[1], 3, 224, 224))).float().cuda()),
            dim=[0]).view(1, -1)
        for i in range(1, len(curr_batch)):
            output = model(torch.from_numpy(curr_batch[i].reshape((curr_batch.shape[1], 3, 224, 224))).float().cuda())
            outputs = torch.cat((outputs, torch.mean(output, dim=[0]).view(1, -1)), dim=0)

        predictions.extend([np.argmax(outputs[i].cpu().data.numpy()) for i in range(outputs.shape[0])])
        idx += len(curr_batch)

    accuracy,classes = evaluate(predictions,labels['validation'])
    print(accuracy)
    print(classes)

if __name__ == '__main__':
    main()
