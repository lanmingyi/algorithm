
# Reinforcement Learning for Solving the Vehicle Routing Problem

We use Reinforcement for solving Travelling Salesman Problem (TSP) and Vehicle Routing Problem (VRP).


## Paper
Implementation of our paper: [Reinforcement Learning for Solving the Vehicle Routing Problem](https://arxiv.org/abs/1802.04240v2). 

## Dependencies


* Numpy
* [tensorflow](https://www.tensorflow.org/)>=1.2
* tqdm

## How to Run
### Train
By default, the code is running in the training mode on a single gpu. For running the code, one can use the following command:
```bash
python main.py --task=vrp10
```

It is possible to add other config parameters like:
```bash
python main.py --task=vrp10 --gpu=0 --n_glimpses=1 --use_tanh=False 
```
There is a full list of all configs in the ``config.py`` file. Also, task specific parameters are available in ``task_specific_params.py``
### Inference
For running the trained model for inference, it is possible to turn off the training mode. For this, you need to specify the directory of the trained model, otherwise random model will be used for decoding:
```bash
python main.py --task=vrp10 --is_train=False --model_dir=./path_to_your_saved_checkpoint
```
The default inference is run in batch mode, meaning that all testing instances are fed simultanously. It is also possible to do inference in single mode, which means that we decode instances one-by-one. The latter case is used for reporting the runtimes and it will display detailed reports. For running the inference with single mode, you can try:
```bash
python main.py --task=vrp10 --is_train=False --infer_type=single --model_dir=./path_to_your_saved_checkpoint
```
### Logs
All logs are stored in ``result.txt`` file stored in ``./logs/task_date_time`` directory.
## Sample CVRP solution

![enter image description here](https://lh3.googleusercontent.com/eUh69ZQsIV4SIE6RjwasAEkdw2VZaTmaeR8Fqk33di70-BGU62fvmcp6HLeGLE61lJDS7jLMpFf2 "Sample VRP")

## Acknowledgements
Thanks to [pemami4911/neural-combinatorial-rl-pytorch](https://github.com/pemami4911/neural-combinatorial-rl-pytorch) for getting the idea of restructuring the code.

## NOTE
1.VRP问题是车辆路径问题的缩写。问题是：有N辆车，都从原点出发，每辆车访问一些点后回到原点，要求所有的点都要被访问到，求最短的车辆行驶距离或最少需要的车辆数或最小化最长行驶距离。

常见的限制要求包括：车辆容量限制、时间窗限制、点访问顺序要求等。


2.车辆路径问题是旅行商问题的推广。在VRP中，目标是为向不同地点交付货物或服务的车队找到最优路线集。VRP最初是由Dantzig和Ramser在1959年提出的。

与TSP类似，VRP也可以用分配给边缘的距离的图来表示。

如果您试图找到一组总距离最小、对车辆没有附加约束的路线，那么最优解决方案是将所有位置分配给一辆车，其余位置空闲。在这种情况下，问题归结为TSP

一个更有趣的问题是最小化所有车辆的最长路线距离的长度，或最长旅行时间的路线的消耗时间。VRPs还可以对车辆有额外的限制—例如，对每辆车访问的地点数量或每条路线的长度的下限和上限。

在以后的文章中，我们还会讲车辆有容量约束和时间窗口的VRP问题。

容量约束：车辆行驶路线上各个地点的总需求不能超过其容量。例如，需求可能是车辆必须交付到每个位置的包裹的大小，其中所有包裹的总大小不能超过车辆的承载能力。

时间窗口：每个位置必须在一个时间窗口[ai, bi]内服务，等待时间是允许的。

位置对之间的优先关系：例如，位置j在位置i之前不能被访问。