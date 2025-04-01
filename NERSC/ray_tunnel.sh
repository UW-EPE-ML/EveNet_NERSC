#!/bin/bash

SCRATCH_REMOTE_PATH="/pscratch/sd/a/avencast/ray_dashboard_info.txt"

echo "[INFO] Waiting for ray_dashboard_info.txt to appear..."
while true; do
  ssh nersc "[ -f $SCRATCH_REMOTE_PATH ]" && break
  sleep 5
done

NODE_PORT=$(ssh nersc "cat $SCRATCH_REMOTE_PATH")
NODE=$(echo $NODE_PORT | awk '{print $1}')
PORT=$(echo $NODE_PORT | awk '{print $2}')

echo "[INFO] Ray is on node $NODE, port $PORT"
echo "[INFO] Setting up SSH tunnel..."

echo
echo "📡 Copy this command to your LOCAL laptop:"
echo "ssh -L ${PORT}:${NODE}:${PORT} nersc"
echo "Then open: http://localhost:${PORT}"
