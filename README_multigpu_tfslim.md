# beeva-poc-distributed-tensorflow
Proof of Concept with Tensorflow & Multi-GPUs at BEEVA Research Lab

### Experiment 2a: multi-GPU on tf-slim

* Training on tensorflow multiple workers, multiple GPU
* Dataset: MNIST. 60000 train samples, 10000 test samples
* Model: LeNet (& Softmax)
* Based on [TF-slim](https://github.com/tensorflow/models/tree/master/slim). Custom [fork](https://github.com/beeva-enriqueotero/models) also.
* Infrastructure: AWS p2.8x (8 gpus). Deep Learning 2.0 AMI. libcudnn.so.5


#### Results:
* num_ps_tasks = 0
* model = lenet

| batch size | gpus (clones) | clone_on_cpu | Accuracy (test) | Steps | Training time (s)
| --- | --- | --- | --- | --- | ---
| 50 | 1 | False | 0.967+-0.001 | 1200 | 40.6+-0.1
| 50 | 2 | False | 0.969 | 1200 | 41.9

#### Conclusions:
* Stable version was not compliant with TF1.0 until 13 march 2017
* We were not able to make it work for multi-worker (replicas)
* No speed-up with multi-gpu (clones) was obtained
* Unsolved [issues](https://github.com/tensorflow/models/issues/1196)

