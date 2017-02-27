# beeva-poc-distributed-tensorflow
Proof of Concept with Distributed Tensorflow at BEEVA Research Lab

### Experiment 1: no GPU
PC: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz, 16GB, 4 processors

#### Deploy

Configure number of steps in `dist_mnist_test.sh`

Launch the dockerized gRPC server
```
sudo docker run tensorflow/tf_grpc_server --cluster_spec="worker|localhost:2222;foo:2222,ps|bar:2222;qux:2222" --job_name=worker --task_id=0
```
Launch the Tensorflow container
```
sudo ./local_test_board.sh https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-1.0.0-cp27-none-linux_x86_64.whl
```
And execute `tensorboard` and `dist_mnist_test.sh` into Tensorflow container
```
# tensorboard browser on localhost:6006
tensorboard --logdir=/tmp/mnist_train/ &
# 3 workers
/var/tf_dist_test/scripts/dist_mnist_test.sh --ps_hosts "localhost:2000,localhost:2001" --worker_hosts "localhost:3000,localhost:3001,localhost:3002" --num_gpus 0
```

#### Results:

| paradigm | workers | Crossentropy | Accuracy | Steps | Training time (s)
| --- | --- | --- | --- | --- | ---
| synch | 3 | 884 | 0.978 | 5000 | 62.8
| synch | 3 | 951 | 0.975 | 5000 | 62.8
| asynch | 3 | 783 | 0.967 | 5000 | 21.6


#### Conclusions: 
* Asynchronous data-parallel is much faster, a little less accurate

