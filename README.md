# Point-cloud regression of geometric primitives

This repository contains the method used in the contest "SHREC 2022: Fitting and recognition of simple geometric primitives on point clouds" by Ivan Sipiran (Department of Computer Science, University of Chile).

Our method consists of two stages: point cloud classification and parameters regression. 

## Networks training
In this repository, we provide the trained networks to perform the task. Nevertheless, here we also describe how we train the models.

### Classification network
The command to train the classification network is:

~~~
python train_classification.py --dataset=<path to training folder> --dataset_type=shrec2022 --outf=<output folder> --feature_transform
~~~

The training loop stores the an intermediate network for each epoch.

### Plane network
The command to train the plane regression network is:

~~~
python train_plane.py --outf=<output folder>
~~~

### Cylinder network
The command to train the cylinder regression network is:

~~~
python train_cylinder.py --outf=<output folder>
~~~

### Sphere network
The command to train the sphere regression network is:

~~~
python train_sphere.py --outf=<output folder>
~~~

### Cone network
The command to train the cone regression network is:

~~~
python train_cone.py --outf=<output folder>
~~~

### Torus network
The command to train the torus regression network is:

~~~
python train_torus.py --outf=<output folder>
~~~

## Execution
We also provide the script to perform the inference for a given point cloud. To perform the inference, the command is:

~~~
python evaluation.py --file=<input file> --outf=<output folder>
~~~

This scripts generates a text file with the same format of the training dataset. It also prints the name of the file and the time required to perform the inference.

To generate the output for the entire test set, we provide a bash file which can be executed with the following command:

~~~
./script.sh 
~~~

You must change the input and output path inside the bash script, according to your dataset.