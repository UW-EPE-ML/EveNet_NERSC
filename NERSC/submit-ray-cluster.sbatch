#!/bin/bash

#SBATCH -C gpu
#SBATCH --time=04:00:00
#SBATCH -q regular
#SBATCH -A m5019_g

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=128

### Give all resources on each node to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128

#SBATCH --reservation=_CAP_EveNet_EPS

head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
fi
port=6379

##############################################################################################

echo srun -n 4 --nodes=4 --ntasks-per-node=1 --gpus-per-task=4 --cpus-per-task=128 --gpus=0 shifter ./start-worker.sh $head_node_ip:$port > ~/ray.run

##############################################################################################

echo "STARTING HEAD at $head_node"
echo "Head node IP: $head_node_ip"
srun --nodes=1 --ntasks=1 --gpus=0 --gpus-per-task=4 --cpus-per-task=128 -w $head_node shifter ./start-head.sh $head_node_ip &
sleep 10

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
srun -n $worker_num --nodes=$worker_num --ntasks-per-node=1 --gpus-per-task=4 --cpus-per-task=128 --gpus=0  --exclude $head_node shifter ./start-worker.sh $head_node_ip:$port &
sleep 5

# export the parent directory of the script to the PYTHONPATH
cd ../ && source setup.sh

shifter python3 evenet/train.py share/pretrain-yulei.yaml

#shifter python3 evenet/train.py downstreams/Quantum/train_TT2L.yaml --load_all

exit