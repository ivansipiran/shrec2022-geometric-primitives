import numpy as np
import torch
import torch.utils.data as data
import os
import glob
from numpy import linalg as LA
import random
import math
import argparse
from model import PointNetCls, PointNetPlane, PointNetCylinder, PointNetSphere, PointNetCone, PointNetTorus
from time import time

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

def origin_mass_center2(pcd):
    expectation = np.mean(pcd, axis = 0)
    centered_pcd = pcd - expectation
    return centered_pcd, expectation

def normalize2(points, unit_ball = False):
    normalized_points, center = origin_mass_center2(points)
    #normalized_points = points
    l2_norm = LA.norm(normalized_points,axis=1)
    max_distance = max(l2_norm)

    if unit_ball:
        scale = max_distance
        normalized_points = normalized_points/(max_distance)
    else:
        scale = 2 * max_distance
        normalized_points = normalized_points/(2 * max_distance)

    return normalized_points, center, scale

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='', help='input file')
parser.add_argument('--outf', type=str, default='', help='output folder')

opt = parser.parse_args()

#generate output filename
filename = opt.file
output_filename = filename.split('/')[-1].split('.')[0]
output_filename = output_filename+'_prediction.txt'

t1 = time()
#Read and sample the point cloud
pcd = np.loadtxt(opt.file, delimiter=',')
pcd = resample_pcd(pcd, 2048)

input_pt, center, scale = normalize2(pcd, unit_ball=True)
#print(f'Center:{center}, Scale: {scale}')
input_pt = torch.unsqueeze(torch.from_numpy(input_pt), 0)

classifier = PointNetCls(k=5, feature_transform=False)
classifier.load_state_dict(torch.load('networks/classification.pth'))
classifier.cuda()

input_pt = input_pt.transpose(2, 1)
input_pt = input_pt.cuda().float()
classifier = classifier.eval()
pred, trans, trans_feat = classifier(input_pt)
pred_choice = pred.detach().max(1)[1].cpu().numpy()[0]

classifier.cpu()

with open(os.path.join(opt.outf, output_filename), 'wt') as f:
    f.write(str(pred_choice+1)+'\n')

    if pred_choice==0: #Plane
        #print('Shape is a plane')
        network = PointNetPlane()
        network.load_state_dict(torch.load("networks/plane.pth"))
        network.cuda()

        network = network.eval()
        pred_normal, pred_point = network(input_pt)
    
        pred_normal = torch.squeeze(pred_normal).cpu().detach().numpy()
        pred_point = torch.squeeze(pred_point).cpu().detach().numpy()

        pred_point = pred_point*scale + center
        mag = LA.norm(pred_normal)
        pred_normal = pred_normal/mag

        f.write(str(pred_normal[0])+'\n')
        f.write(str(pred_normal[1])+'\n')
        f.write(str(pred_normal[2])+'\n')
        f.write(str(pred_point[0])+'\n')
        f.write(str(pred_point[1])+'\n')
        f.write(str(pred_point[2])+'\n')
        
    
        #print(f'Parameters: {pred_normal}->{pred_point}')
    elif pred_choice==1: #Cylinder
        #print('Shape is a cylinder')
        network = PointNetCylinder()
        network.load_state_dict(torch.load("networks/cylinder.pth"))
        network.cuda()

        network = network.eval()
        pred_normal, pred_point, pred_radius = network(input_pt)

        pred_normal = torch.squeeze(pred_normal).cpu().detach().numpy()
        pred_point = torch.squeeze(pred_point).cpu().detach().numpy()
        pred_radius = torch.squeeze(pred_radius).cpu().detach().numpy()

        pred_point = pred_point*scale + center
        mag = LA.norm(pred_normal)
        pred_normal = pred_normal/mag
        pred_radius = pred_radius*scale

        f.write(str(pred_radius)+'\n')
        f.write(str(pred_normal[0])+'\n')
        f.write(str(pred_normal[1])+'\n')
        f.write(str(pred_normal[2])+'\n')
        f.write(str(pred_point[0])+'\n')
        f.write(str(pred_point[1])+'\n')
        f.write(str(pred_point[2])+'\n')

   
        #print(f'Parameters: {pred_normal}->{pred_point}->{pred_radius}')
    elif pred_choice==2: #Sphere
        #print('Shape is a sphere')
        network = PointNetSphere()
        network.load_state_dict(torch.load("networks/sphere.pth"))
        network.cuda()

        network = network.eval()
        pred_point, pred_radius = network(input_pt)

        pred_point = torch.squeeze(pred_point).cpu().detach().numpy()
        pred_radius = torch.squeeze(pred_radius).cpu().detach().numpy()
    
        pred_point = pred_point*scale + center
        pred_radius = pred_radius*scale

        f.write(str(pred_radius)+'\n')
        f.write(str(pred_point[0])+'\n')
        f.write(str(pred_point[1])+'\n')
        f.write(str(pred_point[2])+'\n')

        #print(f'Parameters: {pred_point}->{pred_radius}')
    elif pred_choice==3: #Cone
        #print('Shape is a cone')
        network = PointNetCone()
        network.load_state_dict(torch.load("networks/cone.pth"))
        network.cuda()

        network = network.eval()
        pred_normal, pred_point, pred_aperture = network(input_pt)

        pred_normal = torch.squeeze(pred_normal).cpu().detach().numpy()
        pred_point = torch.squeeze(pred_point).cpu().detach().numpy()
        pred_aperture = torch.squeeze(pred_aperture).cpu().detach().numpy()

        pred_point = pred_point*scale + center
        mag = LA.norm(pred_normal)
        pred_normal = pred_normal/mag

        f.write(str(pred_aperture)+'\n')
        f.write(str(pred_normal[0])+'\n')
        f.write(str(pred_normal[1])+'\n')
        f.write(str(pred_normal[2])+'\n')
        f.write(str(pred_point[0])+'\n')
        f.write(str(pred_point[1])+'\n')
        f.write(str(pred_point[2])+'\n')
    
        #print(f'Parameters: {pred_normal}->{pred_point}->{pred_aperture}')
    elif pred_choice==4: # Torus
        #print('Shape is a torus')
        network = PointNetTorus()
        network.load_state_dict(torch.load("networks/torus.pth"))
        network.cuda()

        network = network.eval()
        pred_normal, pred_point, pred_min, pred_max = network(input_pt)

        pred_normal = torch.squeeze(pred_normal).cpu().detach().numpy()
        pred_point = torch.squeeze(pred_point).cpu().detach().numpy()
        pred_min = torch.squeeze(pred_min).cpu().detach().numpy()
        pred_max = torch.squeeze(pred_max).cpu().detach().numpy()

        pred_point = pred_point*scale + center
        mag = LA.norm(pred_normal)
        pred_normal = pred_normal/mag
        pred_min = pred_min*scale
        pred_max = pred_max*scale

        f.write(str(pred_max)+'\n')
        f.write(str(pred_min)+'\n')
        f.write(str(pred_normal[0])+'\n')
        f.write(str(pred_normal[1])+'\n')
        f.write(str(pred_normal[2])+'\n')
        f.write(str(pred_point[0])+'\n')
        f.write(str(pred_point[1])+'\n')
        f.write(str(pred_point[2])+'\n')

        #print(f'Parameters: {pred_normal}->{pred_point}->{pred_min}->{pred_max}')

t2 = time()
print(f'{output_filename} {(t2-t1)}')