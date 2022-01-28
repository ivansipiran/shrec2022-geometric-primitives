import numpy as np
import torch
import torch.utils.data as data
import os
import glob
from numpy import linalg as LA

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

def origin_mass_center(pcd):
    expectation = np.mean(pcd, axis = 0)
    centered_pcd = pcd - expectation
    return centered_pcd

def normalize(points, unit_ball = False):
    normalized_points = origin_mass_center(points)
    normalized_points = points
    l2_norm = LA.norm(normalized_points,axis=1)
    max_distance = max(l2_norm)

    if unit_ball:
        scale = max_distance
        normalized_points = normalized_points/(max_distance)
    else:
        scale = 2 * max_distance
        normalized_points = normalized_points/(2 * max_distance)

    #return normalized_points, scale
    return normalized_points

class DatasetSHREC2022(data.Dataset):
    def __init__(self, root, npoints=2048, split='train'):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.filepaths = sorted(glob.glob(self.root+'/pointCloud/*.txt'))
        self.filesplit = []
        

        self.objectClass = dict()
        
        for i in range(5):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            with open(gtfile, 'r') as f:
                cl = f.readline()
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            for i in range(5):
                self.filesplit.extend(self.objectClass[i][:7360])
                self.classes.extend([i for j in range(7360)])
        elif self.split == 'val':
            for i in range(5):
                self.filesplit.extend(self.objectClass[i][7360:])
                self.classes.extend([i for j in range(1840)])

    def __len__(self):
        return len(self.filesplit)
    
    def __getitem__(self, idx):
        filename = self.filesplit[idx]

        pcd = np.loadtxt(filename, delimiter=',')
        
        if self.npoints != 0:
            pcd = resample_pcd(pcd, self.npoints)
        
        return self.classes[idx],normalize(pcd, unit_ball=True)

class DatasetPlane(data.Dataset):
    def __init__(self, root, npoints=2048, split='train'):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.filepaths = sorted(glob.glob(self.root+'/pointCloud/*.txt'))
        self.filesplit = []
        

        self.objectClass = dict()
        
        for i in range(5):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            with open(gtfile, 'r') as f:
                cl = f.readline()
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            self.filesplit.extend(self.objectClass[0][:7360])
            #self.classes.extend([i for j in range(7360)])
        elif self.split == 'val':
            self.filesplit.extend(self.objectClass[0][7360:])
            #self.classes.extend([i for j in range(1840)])

    def __len__(self):
        return len(self.filesplit)
    
    def __getitem__(self, idx):
        filename = self.filesplit[idx]

        pcd = np.loadtxt(filename, delimiter=',')
        
        if self.npoints != 0:
            pcd = resample_pcd(pcd, self.npoints)
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        with open(gtfile, 'r') as f:
            cl = f.readline()
            n1 = f.readline()
            n2 = f.readline()
            n3 = f.readline()
        
        normal = np.array([float(n1), float(n2), float(n3)])
        
        return normal,normalize(pcd, unit_ball=True)

if __name__== '__main__':
    input_path = '/media/ivan/a68c0147-4423-4f62-8e54-388f4ace9ec54/Datasets/SHREC2022/dataset/training'
    
    #dataset = DatasetSHREC2022(root=input_path, npoints=2048, split='train')
    #dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True, num_workers=4)
    #cl, target = next(iter(dataloader))
    #print(target.shape)

    dataset = DatasetPlane(root=input_path, npoints=2048, split='val')
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=False, num_workers=0)
    cl, target = next(iter(dataloader))

    print(len(dataset))
    print(cl.shape)
    print(target.shape)