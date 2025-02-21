#!/bin/bash

for i in {2..19}; do
    cp state_space_models/configurations/models/SSM/config1.gin "config${i}.gin"
done
