# beeva-poc-distributed-tensorflow
Proof of Concept with Tensorflow & Multi-GPUs at BEEVA Research Lab

### Experiment 2: multi-GPU

* Training on tensorflow single machine, multiple GPU
* Dataset: MNIST. 60000 train samples, 10000 test samples
* Model: LeNet
* Based on [Transparent multi-gpu training on Tensorflow with Keras](https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012#.w0nbus9yu). Custom [fork](https://github.com/beeva-enriqueotero/keras-extras/blob/master/examples/mnist_cnn_multi.py) to implement example and fix TF 1.0 compatibility
* ***Note**: first (failed) attempt was using tf-slim. [More info](https://github.com/tensorflow/models/issues/1196)*
* Infrastructure: AWS p2.8x (8 gpus). Deep Learning 2.0 AMI. libcudnn.so.5

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

| batch size | gpus | Accuracy (test) | Epochs | Training time (s/epoch)
| --- | --- | --- | --- | ---
| 128 | 1 | 0.9884 | 12 | 6.8
| 128 | 2 | 0.9898 | 12 | 5.2
| 128 | 3 | error | error | error
| 128 | 4 | 0.9891 | 12 | 4.9
| 128 | 8 | 0.9899 | 12 | 6.4

#### Conclusions: 
* 4 gpus is the fastests configuration. Only 30% faster than 1 gpu
* 8 gpus is slower than 4 
* Due to technical implementation details, only even number of gpus allowed


