# beeva-poc-distributed-tensorflow
Proof of Concept with Tensorflow & Multi-GPUs at BEEVA Research Lab

### Experiment 2: multi-GPU

* Training on tensorflow single machine, multiple GPU
* Dataset: MNIST. 60000 train samples, 10000 test samples
* Model: Simple Convnet (5 layers) inspired by LeNet
* Based on [Transparent multi-gpu training on Tensorflow with Keras](https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012#.w0nbus9yu). Custom [fork](https://github.com/beeva-enriqueotero/keras-extras/blob/master/examples/mnist_cnn_multi.py) to implement example and fix TF 1.0 compatibility
* ***Note**: first (failed) attempt was using tf-slim. [More info](README_multigpu_tfslim.md)*
* Infrastructure 1: AWS p2.8x (8 gpus). Deep Learning 2.0 AMI. libcudnn.so.5
* Infrastructure 2: Google n1-standard-16 with 2 gpus (2 x nVidia Tesla K80), tensorflow-gpu==1.01, Keras==2.0.2, NVIDIA Driver 375.39, No CuDNN

#### Deploy

[*Optional*] Modify print time format on Keras `generic_utils.py`
```
sudo nano /usr/lib/python2.7/dist-packages/Keras-1.2.2-py2.7.egg/keras/utils/generic_utils.py
```
Clone `keras-extras`
```
git clone https://github.com/beeva-enriqueotero/keras-extras
```
Launch multi-gpu experiment
```
time python keras-extras/examples/mnist_cnn_multi.py  --extras /home/ec2-user/keras-extras/ --gpus 2
```

#### Results:

| infrastructure | batch size | gpus | Accuracy (test) | Epochs | Training time (s/epoch)
| --- | --- | --- | --- | --- | ---
| 1 | 128 | 1 | 0.9884 | 12 | 6.8
| 1 | 128 | 2 | 0.9898 | 12 | 5.2
| 1 | 128 | 3 | error | error | error
| 1 | 128 | 4 | 0.9891 | 12 | 4.9
| 1 | 128 | 8 | 0.9899 | 12 | 6.4
| 2 | 128 | 2 | error | 12 | error


#### Conclusions: 
* 4 gpus is the fastests configuration. Only 30% faster than 1 gpu
* 8 gpus is slower than 4 
* Due to technical implementation details, only even number of gpus allowed
* Google Engine Documentation about [attaching GPUs to instances](https://cloud.google.com/compute/docs/gpus/add-gpus) doesn't include references to CuDNN

