#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Choose a dashboard port (or randomize it safely)
PORT=8265

echo "Ray will start on $(hostname) at port $PORT" > "$SCRATCH"/ray_dashboard_info.txt
# Launch the head node
ray start --head --dashboard-host 0.0.0.0 --port=6379 --dashboard-port=$PORT
sleep infinity