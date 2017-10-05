# beeva-poc-distributed-tensorflow
Proof of Concept with Tensorflow & Multi-GPUs at BEEVA Research Lab

### Experiment: multi-GPU

* Training on tensorflow single machine, multiple GPU
* Dataset: CIFAR10. 50000 train samples, 10000 test samples
* Models: [Alexnet variant](https://www.tensorflow.org/tutorials/deep_cnn#training_a_model_using_multiple_gpu_cards), [Resnet50](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator) 
* **Infrastructure 1**: AWS p2.8x (8 gpus nvidia Tesla K80). Deep Learning 2.3 Ubuntu AMI. Tensorflow, nvidia driver 375.66, CUDA 8.0, libcudnn.so.5.1.10

#### Deploy

#### Run

Launch multi-gpu cifar10 experiment (old)
```
python models/tutorials/image/cifar10/cifar10_multi_gpu_train.py
```

Launch multi-gpu cifar10_estimator experiment
```
# python models/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.py --data-dir=${PWD}/cifar-10-data

python tutorials/image/cifar10_estimator/cifar10_main.py --data-dir=${PWD}/cifar-10-data --job-dir=/tmp/cifar10 --num-gpus=1 --num-layers=50 --train-batch-size=128 --train-steps=4700 --learning-rate=0.1 --variable-strategy=GPU

python models/tutorials/image/cifar10_estimator/cifar10_main.py --data-dir=${PWD}/cifar-10-data --job-dir=/tmp/cifar10 --num-gpus=8 --num-layers=50 --train-batch-size=2048 --train-steps=290 --learning-rate=0.1 --variable-strategy=GPU --eval-batch-size=400
```



#### Results:

| infrastructure | model | batch size | gpus | Accuracy (validation) | Epochs | Throughput | GPU util
| --- | --- | --- | --- | --- | --- | --- | ---
| 1 | AlexNet | 128 | 1 | ? | ? | 4300 | ? 
| 1 | AlexNet | 64 | 1 | ? | ? | 4300 | ?
| 1 | AlexNet | 64 | 8 | ? | ? | 19000 | 55%
| 1 | AlexNet | 128 | 8 | ? | ? | 20000 | 63%
| 1 | AlexNet | 128 | 4 | ? | ? | 16000 | 93%
| 1 | AlexNet | 128 | 2 | ? | ? | 8500 | 94%
| 1 | ResNet50 | 128 (lr=0.1) | 1 | 0.657 | 12 (4700 steps) | 700 | 80%
| 1 | ResNet50 | 256 | 1 | 0.6872 | 12 | 800 | 90%
| 1 | ResNet50 | 512 | 1 | x | x | 900 | 95%
| 1 | ResNet50 | 8x256 (lr=0.1)| 1 | 0.11 | 12 (300) | 5000 | 90%
| 1 | ResNet50 | 8x256 (lr=0.4)| 1 | 0.16 | 12 (300) | 5000 | 90%
