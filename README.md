# beeva-poc-distributed-tensorflow
Proof of Concept with Distributed Tensorflow at BEEVA Research Lab

### Experiment 1: no GPU
PC: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz, 16GB, 4 processors

### Deploy

```
sudo docker run tensorflow/tf_grpc_server --cluster_spec="worker|localhost:2222;foo:2222,ps|bar:2222;qux:2222" --job_name=worker --task_id=0
# 3 workers
/var/tf_dist_test/scripts/dist_mnist_test.sh --ps_hosts "localhost:2000,localhost:2001" --worker_hosts "localhost:3000,localhost:3001,localhost:3002" --num_gpus 0
```

#### Results:

| paradigm | workers | Crossentropy | Accuracy | Training time (s)
| --- | -----------| ---- | ---- | ----
| synch | 3 | 884 | 0.978 | 62.8
| synch | 3 | 951 | 0.975 | 62.8
| asynch | 3 | 783 | 0.967 | 21.6


#### Conclusions: 
* Asynchronous data-parallel is much faster


