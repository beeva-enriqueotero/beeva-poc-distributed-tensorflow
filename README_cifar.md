# beeva-poc-distributed-tensorflow
Proof of Concept with Tensorflow & Multi-GPUs at BEEVA Research Lab

### Experiment: multi-GPU

* Training on tensorflow single machine, multiple GPU
* Dataset: CIFAR10. 60000 train samples, 10000 test samples
* Model: Alexnet variant 
* **Infrastructure 1**: AWS p2.8x (8 gpus nvidia Tesla K80). Deep Learning 2.3 Ubuntu AMI. Tensorflow, nvidia driver 375.66, CUDA 8.0, libcudnn.so.5.1.10

#### Deploy

#### Run

Launch multi-gpu experiment
```
python models/tutorials/image/cifar10/cifar10_multi_gpu_train.py
```

#### Results:

| [infrastructure](https://github.com/beeva-enriqueotero/beeva-poc-distributed-tensorflow/blob/master/README_multigpu.md#experiment-2-multi-gpu) | batch size | gpus | Accuracy (validation) | Epochs | Throughput (s/epoch)
| --- | --- | --- | --- | --- | --- | ---
| 1 | 128 | 1 | ? | ? | 4300 | ? 
| 1 | 64 | 1 | ? | ? | 4300 | ?
| 1 | 64 | 8 | ? | ? | 19000 | 55%
| 1 | 128 | 8 | ? | ? | 20000 | 63%
