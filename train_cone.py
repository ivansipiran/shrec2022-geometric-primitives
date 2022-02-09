from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import DatasetCone
from model import PointNetCone
import torch.nn.functional as F
from tqdm import tqdm
import visdom
import numpy as np
import matplotlib.pyplot as plt

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
#parser.add_argument('--dataset', type=str, required=True, help="dataset path")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = DatasetCone(
        root="/media/ivan/a68c0147-4423-4f62-8e54-388f4ace9ec54/Datasets/SHREC2022/dataset/training",
        npoints=opt.num_points,
        split='train', transform=False)

test_dataset = DatasetCone(
        root="/media/ivan/a68c0147-4423-4f62-8e54-388f4ace9ec54/Datasets/SHREC2022/dataset/training",
        split='val',
        npoints=opt.num_points, transform=False)

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

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCone()

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()
myloss = torch.nn.MSELoss()
mylossCen = torch.nn.MSELoss()
mylossA = torch.nn.MSELoss()


num_batch = len(dataset) / opt.batchSize

lossTrainValues = []
lossTestValues = []
lossLoss1 = []
lossLoss2 = []
lossLoss3 = []
lossLoss4 = []
delta = 1/256

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        target_normal, target_vertex, target_aperture, points = data
        points = points.transpose(2, 1)
        points, target_normal = points.cuda().float(), target_normal.cuda().float()
        target_vertex = target_vertex.cuda().float()
        target_aperture = target_aperture.cuda().float()
        
        optimizer.zero_grad()
        classifier = classifier.train()
        pred_normal, pred_vertex, pred_aperture = classifier(points)
        
        lossCos = (1.0 - torch.pow(F.cosine_similarity(pred_normal, target_normal),8)) 
        lossL2 = myloss(pred_normal, target_normal)
        lossCen = mylossCen(pred_vertex, target_vertex)
        lossAper = mylossA(pred_aperture, target_aperture)
        loss = lossCos + lossL2 + lossCen + lossAper
        loss.mean().backward()
        optimizer.step()
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.mean().item()))

        lossTrainValues.append(loss.mean().item())
        vis_curve(lossTrainValues, "train", "train", vis)

    #Validation after one epoch
    running_loss = 0
    running_cos = 0
    running_l2 = 0
    running_ver = 0
    running_aper = 0

    cont = 0
    for i,data in tqdm(enumerate(testdataloader, 0)):
        target_normal, target_vertex, target_aperture, points = data
        points = points.transpose(2, 1)
        points, target_normal = points.cuda().float(), target_normal.cuda().float()
        target_vertex = target_vertex.cuda().float()
        target_aperture = target_aperture.cuda().float()

        classifier = classifier.eval()
        
        pred_normal, pred_vertex, pred_aperture = classifier(points)

        lossCos = (1.0 - torch.pow(F.cosine_similarity(pred_normal, target_normal),8)) #+ delta*torch.pow(radius*scale - r, 2)
        lossL2 = myloss(pred_normal, target_normal)
        lossCen = mylossCen(pred_vertex, target_vertex)
        lossAper = mylossA(pred_aperture, target_aperture)
        loss = lossCos + lossL2 + lossCen + lossAper

        running_loss += loss.item()
        running_cos += lossCos.item()
        running_l2 += lossL2.item()
        running_ver += lossCen.item()
        running_aper += lossAper.item()

        cont = cont + 1
    
    lossTestValues.append(running_loss/float(cont))
    lossLoss1.append(running_cos/float(cont))
    lossLoss2.append(running_l2/float(cont))
    lossLoss3.append(running_ver/float(cont))
    lossLoss4.append(running_aper/float(cont))

    
    vis_curve(lossTestValues, "test", "test", vis)
    vis_curve(lossLoss1, "cos", "cos", vis)
    vis_curve(lossLoss2, "l2", "l2", vis)
    vis_curve(lossLoss3, "vertex", "vertex", vis)
    vis_curve(lossLoss4, "aperture", "aperture", vis)

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))


angle_err = 0
point_err = 0
ape_err = 0

angles = []
distances = []
apertures = []

cont = 0

for i,data in tqdm(enumerate(testdataloader, 0)):
    target_normal, target_center, target_aperture, points = data
    points = points.transpose(2, 1)
    points, target_normal = points.cuda().float(), target_normal.cuda().float()
    target_center, target_aperture = target_center.cuda().float(), target_aperture.cuda().float()

    classifier = classifier.eval()
    pred_normal, pred_center, pred_aperture = classifier(points)
    
    t = np.squeeze(target_normal.detach().cpu().numpy())
    p = np.squeeze(pred_normal.detach().cpu().numpy()) 
    norm_p = np.linalg.norm(p)
    p = p/norm_p
    angle = 180*np.arccos(t.dot(p))/np.pi
    angles.append(angle)
    angle_err += angle

    t1 = np.squeeze(target_center.detach().cpu().numpy())
    p1 = np.squeeze(pred_center.detach().cpu().numpy()) 
    dist = np.linalg.norm(t1-p1)
    distances.append(dist)
    point_err += dist

    t2 = np.squeeze(target_aperture.detach().cpu().numpy())
    p2 = np.squeeze(pred_aperture.detach().cpu().numpy()) 
    dist2 = np.linalg.norm(t2-p2)
    apertures.append(dist2)
    ape_err += dist2

    print(f'{t} -> {p}->{angle}')
    cont = cont + 1
    
print("average angle error {}".format(angle_err / float(cont)))
print("average point error {}".format(point_err / float(cont)))
print("average radii error {}".format(ape_err / float(cont)))


fig,axes = plt.subplots(1,3)
axes[0].hist(angles, 50)
axes[1].hist(distances, 50)
axes[2].hist(apertures, 50)
plt.show()
