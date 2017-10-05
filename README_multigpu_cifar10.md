# beeva-poc-distributed-tensorflow
Proof of Concept with Tensorflow & Multi-GPUs at BEEVA Research Lab

### Experiment: multi-GPU

* Training on tensorflow single machine, multiple GPU
* Dataset: CIFAR10. 50000 train samples, 10000 test samples
* Models: [Alexnet variant](https://www.tensorflow.org/tutorials/deep_cnn#training_a_model_using_multiple_gpu_cards), [Resnet50](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator) 
* **Infrastructure 1**: AWS p2.8x (8 gpus nvidia Tesla K80). Deep Learning 2.3 Ubuntu AMI. Tensorflow, nvidia driver 375.66, CUDA 8.0, libcudnn.so.5.1.10

#### Deploy

#### Run

Launch multi-gpu experiment
```
python models/tutorials/image/cifar10/cifar10_multi_gpu_train.py
python models/tutorials/image/cifar10_estimator/cifar10_main.py --data-dir=${PWD}/cifar-10-data --job-dir=/tmp/cifar10 --num-gpus=1 --num-layers=50
```

#### Results:

| infrastructure | model | batch size | gpus | Accuracy (validation) | Epochs | Throughput (s/epoch) | GPU util
| --- | --- | --- | --- | --- | --- | --- | ---
| 1 | AlexNet | 128 | 1 | ? | ? | 4300 | ? 
| 1 | AlexNet | 64 | 1 | ? | ? | 4300 | ?
| 1 | AlexNet | 64 | 8 | ? | ? | 19000 | 55%
| 1 | AlexNet | 128 | 8 | ? | ? | 20000 | 63%
| 1 | AlexNet | 128 | 4 | ? | ? | 16000 | 93%
| 1 | AlexNet | 128 | 2 | ? | ? | 8500 | 94%
| 1 | ResNet50 | x | 1 | x | x | x | x
| 1 | ResNet50 | 128 | 1 | x | x | 700 | 80%
| 1 | ResNet50 | 256 | 1 | x | x | 800 | 90%
| 1 | ResNet50 | 512 | 1 | x | x | 900 | 95%
