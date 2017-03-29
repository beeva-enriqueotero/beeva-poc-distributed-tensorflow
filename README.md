# beeva-poc-distributed-tensorflow
Proof of Concept with Distributed Tensorflow at BEEVA Research Lab

### Experiment 1a: no GPU

* MNIST training on dockerized distributed tensorflow locally
* Based on https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/dist_test
* Custom [fork](https://github.com/beeva-enriqueotero/tensorflow/blob/master/tensorflow/tools/dist_test/python/mnist_replica.py) to output accuracy and use tensorboard
* PC: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz, 16GB, 4 processors

![Tensorboard](/images/tensorboard_mnist.png)

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


### Experiment 1b: with GPUs

* MNIST training on dockerized distributed tensorflow locally
* Based on https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/dist_test
* Custom [fork](https://github.com/beeva-enriqueotero/tensorflow/blob/master/tensorflow/tools/dist_test/python/mnist_replica.py) to output accuracy and use tensorboard
* PC: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz, 16GB, 4 processors

#### Deploy:
Edit Dockerfile.
```
COPY tensorflow_*.whl /
RUN pip install /tensorflow_*.whl

```
Generate gcloud json key
Execute in a root console
```
export TF_DIST_CONTAINER_CLUSTER="test-cluster-1"
export TF_DIST_GCLOUD_KEY_FILE="my-gcloud-key.json"
export TF_DIST_GCLOUD_COMPUTE_ZONE="europe-west1-b"
export TF_DIST_GCLOUD_PROJECT="poc-tensorflow-cloud-ml"
./remote_test.sh https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp27-none-linux_x86_64.whl
```
Alternative with `--setup_cluster_only`

#### Results:
```
ERROR: (gcloud.container.clusters.get-credentials) ResponseError: code=404, message=The resource "projects/poc-tensorflow-cloud-ml/zones/europe-west1-b/clusters/test-cluster-1" was not found.
No cluster named 'test-cluster-1' in poc-tensorflow-cloud-ml.
FAILED to get credentials for container cluster: test-cluster-1
FAILED to determine GRPC server URLs of all workers
```
#### Conclusions: 
* We got errors trying to deploy ./remote_test.sh GCE
* We don't know how to deploy ./remote_test.sh on AWS
* We didn't found documentation about running on Google Container Engine with GPUs

### [Experiment 2: multi-GPU](README_multigpu.md)
