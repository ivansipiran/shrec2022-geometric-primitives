import numpy as np
import torch
import torch.utils.data as data
import os
import glob
from numpy import linalg as LA
import random
import math

def get_rotation_x(teta):
    return np.array([
        np.array([1,    0,                0]),
        np.array([0,    math.cos(teta),   -math.sin(teta)]),
        np.array([0,    math.sin(teta),    math.cos(teta)])
    ])

def get_rotation_y(teta):
    return np.array([
        np.array([math.cos(teta),  0,       math.sin(teta)]),
        np.array([0,               1,       0]),
        np.array([-math.sin(teta), 0,       math.cos(teta)])
    ])


def get_rotation_z(teta):
    return np.array([
        np.array([math.cos(teta),  -math.sin(teta),       0]),
        np.array([math.sin(teta),  math.cos(teta),        0]),
        np.array([0,               0,                     1])
    ])


def add_rotation_to_pcloud(pcloud, r_rotation):
    # r_rotation = rand_rotation_matrix()
    if len(pcloud.shape) == 2:
        return pcloud.dot(r_rotation)
    else:
        return np.asarray([e.dot(r_rotation) for e in pcloud])

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

def origin_mass_center2(pcd):
    expectation = np.mean(pcd, axis = 0)
    centered_pcd = pcd - expectation
    return centered_pcd, expectation

def normalize(points, unit_ball = False):
    normalized_points = origin_mass_center(points)
    #normalized_points = points
    l2_norm = LA.norm(normalized_points,axis=1)
    max_distance = max(l2_norm)

    if unit_ball:
        scale = max_distance
        normalized_points = normalized_points/(max_distance)
    else:
        scale = 2 * max_distance
        normalized_points = normalized_points/(2 * max_distance)

    return normalized_points, scale
    
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

# Dataset class for the classification problem    
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
        
        norm_points, center, scale = normalize2(pcd, unit_ball=True)
        
        return self.classes[idx],norm_points

#Dataset class for the plane regression
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
        elif self.split == 'val':
            self.filesplit.extend(self.objectClass[0][7360:])
            
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

#Dataset class for the cylinder regression
class DatasetCylinder(data.Dataset):
    def __init__(self, root, npoints=2048, split='train'):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.filepaths = [os.path.normpath(fi) for fi in sorted(glob.glob(self.root+'/pointCloud/*.txt'))]
        self.filesplit = []
        
        print(len(self.filepaths))
        self.objectClass = dict()
        
        for i in range(5):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gtfile = os.path.normpath(gtfile)
            
            with open(gtfile, 'r') as f:
                cl = f.readline()
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            self.filesplit.extend(self.objectClass[1][:7360])
        elif self.split == 'val':
            self.filesplit.extend(self.objectClass[1][7360:])
            
    def __len__(self):
        return len(self.filesplit)
    
    def __getitem__(self, idx):
        filename = self.filesplit[idx]

        pcd = np.loadtxt(filename, delimiter=',')
        
        if self.npoints != 0:
            pcd = resample_pcd(pcd, self.npoints)
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        gtfile = os.path.normpath(gtfile)

        with open(gtfile, 'r') as f:
            cl = f.readline()
            radius = f.readline()
            n1 = f.readline()
            n2 = f.readline()
            n3 = f.readline()
            c1 = f.readline()
            c2 = f.readline()
            c3 = f.readline()
            
        
        target_normal = np.array([float(n1), float(n2), float(n3)])
        point = np.array([float(c1), float(c2), float(c3)])
        radius = np.float(radius)

        norm_points, center, scale = normalize2(pcd, unit_ball=True)
        target_point = (point - center)/scale
        radius = radius / scale
        
        return target_normal, target_point, radius, norm_points

#Dataset class for the cone regression
class DatasetCone(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', transform=True):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.transform = transform
        self.filepaths = [os.path.normpath(fi) for fi in sorted(glob.glob(self.root+'/pointCloud/*.txt'))]
        self.filesplit = []
        
        print(len(self.filepaths))
        self.objectClass = dict()
        
        for i in range(5):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gtfile = os.path.normpath(gtfile)
            
            with open(gtfile, 'r') as f:
                cl = f.readline()
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            self.filesplit.extend(self.objectClass[3][:7360])
        elif self.split == 'val':
            self.filesplit.extend(self.objectClass[3][7360:])
            
    def __len__(self):
        return len(self.filesplit)
    
    def __getitem__(self, idx):
        filename = self.filesplit[idx]

        pcd = np.loadtxt(filename, delimiter=',')
        
        if self.npoints != 0:
            pcd = resample_pcd(pcd, self.npoints)
        
        #Apply a perturbation in rotation
        if self.transform:
            rot_x = get_rotation_x(np.deg2rad(random.uniform(25, 45)))
            rot_y = get_rotation_y(np.deg2rad(random.uniform(25, 45)))
            rot_z = get_rotation_z(np.deg2rad(random.uniform(25, 45)))
            rotation_mat = np.dot(rot_x, rot_y)
            rotation_mat = np.dot(rotation_mat, rot_z)
        
            pcd = add_rotation_to_pcloud(pcd, rotation_mat)
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        gtfile = os.path.normpath(gtfile)

        with open(gtfile, 'r') as f:
            cl = f.readline()
            aperture = f.readline()
            n1 = f.readline()
            n2 = f.readline()
            n3 = f.readline()
            v1 = f.readline()
            v2 = f.readline()
            v3 = f.readline()
        
        normal = np.array([float(n1), float(n2), float(n3)])
        vertex = np.array([float(v1), float(v2), float(v3)])
        aperture = np.float(aperture)

        if self.transform:
            rotation_norm = np.transpose(np.linalg.inv(rotation_mat))
        
            normal = np.dot(rotation_norm, normal)
            normal = normal/np.linalg.norm(normal)

        norm_points, center, scale = normalize2(pcd, unit_ball=True)
        target_vertex = (vertex-center)/scale
        
        return normal, target_vertex, aperture, norm_points

#Dataset class for the sphere regression
class DatasetSphere(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', transform=True):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.transform = transform
        self.filepaths = [os.path.normpath(fi) for fi in sorted(glob.glob(self.root+'/pointCloud/*.txt'))]
        self.filesplit = []
        
        print(len(self.filepaths))
        self.objectClass = dict()
        
        for i in range(5):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gtfile = os.path.normpath(gtfile)
            
            with open(gtfile, 'r') as f:
                cl = f.readline()
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            self.filesplit.extend(self.objectClass[2][:7360])
        elif self.split == 'val':
            self.filesplit.extend(self.objectClass[2][7360:])
            
    def __len__(self):
        return len(self.filesplit)
    
    def __getitem__(self, idx):
        filename = self.filesplit[idx]

        pcd = np.loadtxt(filename, delimiter=',')
        
        if self.npoints != 0:
            pcd = resample_pcd(pcd, self.npoints)
        
        #Apply a perturbation in rotation
        if self.transform:
            rot_x = get_rotation_x(np.deg2rad(random.uniform(25, 45)))
            rot_y = get_rotation_y(np.deg2rad(random.uniform(25, 45)))
            rot_z = get_rotation_z(np.deg2rad(random.uniform(25, 45)))
            rotation_mat = np.dot(rot_x, rot_y)
            rotation_mat = np.dot(rotation_mat, rot_z)
        
            pcd = add_rotation_to_pcloud(pcd, rotation_mat)
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        gtfile = os.path.normpath(gtfile)

        with open(gtfile, 'r') as f:
            cl = f.readline()
            radius = f.readline()
            c1 = f.readline()
            c2 = f.readline()
            c3 = f.readline()
        
        center = np.array([float(c1), float(c2), float(c3)])
        radius = np.float(radius)

        if self.transform:
            rotation_norm = np.transpose(np.linalg.inv(rotation_mat))
        
            normal = np.dot(rotation_norm, normal)
            normal = normal/np.linalg.norm(normal)

        norm_points, centerp, scale = normalize2(pcd, unit_ball=True)
        target = (center - centerp)/scale
        radius = radius / scale
        
        return target, radius, norm_points

#Dataset class for the torus regression
class DatasetTorus(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', transform=True):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.transform = transform
        self.filepaths = [os.path.normpath(fi) for fi in sorted(glob.glob(self.root+'/pointCloud/*.txt'))]
        self.filesplit = []
        
        print(len(self.filepaths))
        self.objectClass = dict()
        
        for i in range(5):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gtfile = os.path.normpath(gtfile)
            
            with open(gtfile, 'r') as f:
                cl = f.readline()
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            self.filesplit.extend(self.objectClass[4][:7360])
        elif self.split == 'val':
            self.filesplit.extend(self.objectClass[4][7360:])
            
    def __len__(self):
        return len(self.filesplit)
    
    def __getitem__(self, idx):
        filename = self.filesplit[idx]

        pcd = np.loadtxt(filename, delimiter=',')
        
        if self.npoints != 0:
            pcd = resample_pcd(pcd, self.npoints)
        
        #Apply a perturbation in rotation
        if self.transform:
            rot_x = get_rotation_x(np.deg2rad(random.uniform(25, 45)))
            rot_y = get_rotation_y(np.deg2rad(random.uniform(25, 45)))
            rot_z = get_rotation_z(np.deg2rad(random.uniform(25, 45)))
            rotation_mat = np.dot(rot_x, rot_y)
            rotation_mat = np.dot(rotation_mat, rot_z)
        
            pcd = add_rotation_to_pcloud(pcd, rotation_mat)
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        gtfile = os.path.normpath(gtfile)

        with open(gtfile, 'r') as f:
            cl = f.readline()
            major_radius = f.readline()
            minor_radius = f.readline()
            n1 = f.readline()
            n2 = f.readline()
            n3 = f.readline()
            c1 = f.readline()
            c2 = f.readline()
            c3 = f.readline()
        
        normal = np.array([float(n1), float(n2), float(n3)])
        center = np.array([float(c1), float(c2), float(c3)])
        major_radius = np.float(major_radius)
        minor_radius = np.float(minor_radius)

        if self.transform:
            rotation_norm = np.transpose(np.linalg.inv(rotation_mat))
        
            normal = np.dot(rotation_norm, normal)
            normal = normal/np.linalg.norm(normal)

        norm_points, centerp, scale = normalize2(pcd, unit_ball=True)
        target_point = (center - centerp)/scale
        target_radius_min = minor_radius/scale
        target_radius_max = major_radius/scale
        
        return normal, target_point, target_radius_min, target_radius_max, norm_points

if __name__== '__main__':
    input_path = "/media/ivan/a68c0147-4423-4f62-8e54-388f4ace9ec54/Datasets/SHREC2022/dataset/training"

    #Tests
    
    #Entire dataset    
    #dataset = DatasetSHREC2022(root=input_path, npoints=2048, split='train')
    #dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True, num_workers=4)
    #print(len(dataset.classes))
    #cl, target = next(iter(dataloader))
    #print(target.shape)

    #Plane dataset
    #dataset = DatasetPlane(root=input_path, npoints=2048, split='val')
    #dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=False, num_workers=0)
    #cl, target = next(iter(dataloader))

    #Cylinder dataset
    #dataset = DatasetCylinder(root=input_path, npoints=2048, split='train')
    #dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=False, num_workers=0)
    #print(len(dataset))
    #rad, scale, norm, target = next(iter(dataloader))

    #Cone dataset
    #dataset = DatasetCone(root=input_path, npoints=2048, split='train')
    #dataloader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=False, num_workers=0)
    #print(len(dataset))
    #norm, target = next(iter(dataloader))

    #Torus dataset
    #dataset = DatasetTorus(root=input_path, npoints=2048, split='val', transform=False)
    #dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False, num_workers=0)
    #print(len(dataset))
    #norm, minR, maxR, center, scale, target = next(iter(dataloader))

    #Sphere dataset
    #dataset = DatasetSphere(root=input_path, npoints=2048, split='val', transform=False)
    #dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False, num_workers=0)
    #print(len(dataset))
    #center, radius, points = next(iter(dataloader))
