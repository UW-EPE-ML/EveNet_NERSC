#!/bin/bash

salloc --nodes 4 --qos interactive --time 04:00:00 --constraint gpu --gpus 16 --account m2616_g --image=registry.nersc.gov/m2616/avencast/evenet:1.0