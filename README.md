# Anomaly_analysis

## Build and run the environment
To build docker image with the environment simply run:
```bash
./build_docker.sh
```

Once created, one can run a container in interactive mode typing:
```bash
./run_docker.sh $gpu_id
```
where ID of a GPU -- `$gpu_id` has to be specified with regards to `docker run` e.g. 0, 1, 2, etc. 

Once navigating in the container type:
```bash
cd src/
```

## Data unpacking

Some of the datasets were compressed due to github's storage limitations. To unpack them run:

```bash
unzip data/ccfd.zip
tar -xf data/census-income-full-mixed-binarized.tar.xz -C data/
```

## Running models

Navigate to `experiments/` directory:

```bash
cd experiments/
```

### Tabular datasets

To run experiment for one of the tabular datasets run:

```bash
python ./tabular.py -d $dataset -o $output_path -m $model -n $n_times --gamma $gamma
```

where:

- `$dataset` is the name of desired dataset. One of: `ccfd` | `kddps` | `kddh` | `celeba` | `census` | `campaign` | `thyroid`.
- `$output_path` is the path to csv with resulting AUCs. By default `result.txt`.
- `$model` is the model name. One of: `e2econva3` | `a3`. By default `e2econva3`.
- `$n_times` is the number if trial runs. By default 1.
- `$gamma` is the $\gamma$ parameter. By default 1.

### Image datasets

Similarily to tabular datasets run:

```bash
python ./image.py -d $dataset -c $normal_cls -o $output_path -n $n_times --gamma $gamma
```

All of the arguments are the same as above except for:

- `$dataset` is one of: `cifar10` | `mnist` | `fmnist`.
- `$normal_cls` is the selected normal class index. By default 0.
