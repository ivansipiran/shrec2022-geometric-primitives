from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import DatasetPlane
from model import PointNetPlane
import torch.nn.functional as F
from tqdm import tqdm
import visdom
import numpy as np

def vis_curve(curve, window, name, vis):
    vis.line(X=np.arange(len(curve)),
                 Y=np.array(curve),
                 win=window,
                 opts=dict(title=name, legend=[name + "_curve"], markersize=2, ), )

vis = visdom.Visdom(port = 8997, env="TRAIN")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2048, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = DatasetPlane(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train')

test_dataset = DatasetPlane(
        root=opt.dataset,
        split='val',
        npoints=opt.num_points)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
#num_classes = len(dataset.classes)
#print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetPlane()

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()


num_batch = len(dataset) / opt.batchSize

lossTrainValues = []
lossTestValues = []

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        target, points = data
        #target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda().float(), target.cuda().float()
        #print(target)
        optimizer.zero_grad()
        classifier = classifier.train()
        pred = classifier(points)
        #print(pred)
        loss = 1.0 - torch.pow(F.cosine_similarity(pred, target),9)
        #print(loss)
        loss.mean().backward()
        optimizer.step()
        #pred_choice = pred.data.max(1)[1]
        #correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.mean().item()))

        lossTrainValues.append(loss.mean().item())
        vis_curve(lossTrainValues, "train", "train", vis)

        #if i % 10 == 0:
        #    j, data = next(enumerate(testdataloader, 0))
        #    target, points = data
            #target = target[:, 0]
        #    points = points.transpose(2, 1)
        #    points, target = points.cuda().float(), target.cuda()
        #    classifier = classifier.eval()
        #    pred, _, _ = classifier(points)
        #    loss = F.nll_loss(pred, target)
        #    pred_choice = pred.data.max(1)[1]
         #   correct = pred_choice.eq(target.data).cpu().sum()
         #   print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
    
    #Validation after one epoch
    running_loss = 0
    cont = 0
    for i,data in tqdm(enumerate(testdataloader, 0)):
        target, points = data
        #target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda().float(), target.cuda().float()
        classifier = classifier.eval()
        pred = classifier(points)
        loss = 1.0 - torch.pow(F.cosine_similarity(pred, target),9)
        running_loss += loss.item()
        cont = cont + 1
    
    lossTestValues.append(running_loss/float(cont))
    vis_curve(lossTestValues, "test", "test", vis)

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

#total_correct = 0
#total_testset = 0
running_loss = 0
cont = 0

for i,data in tqdm(enumerate(testdataloader, 0)):
    target, points = data
    #target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda().float(), target.cuda().float()
    classifier = classifier.eval()
    pred = classifier(points)
    t = np.squeeze(target.detach().cpu().numpy())
    p = np.squeeze(pred.detach().cpu().numpy())
    norm_p = np.linalg.norm(p)
    p = p/norm_p
    angle = 180*np.arccos(t.dot(p))/np.pi
    print(f'{t} -> {p}->{angle}')
    loss = 1.0 - torch.pow(F.cosine_similarity(pred, target),9)
    running_loss += loss.item()
    cont = cont + 1
    #pred_choice = pred.data.max(1)[1]
    #correct = pred_choice.eq(target.data).cpu().sum()
    #total_correct += correct.item()
    #total_testset += points.size()[0]

print("final accuracy {}".format(running_loss / float(cont)))
print(cont)