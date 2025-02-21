#!/bin/bash

for i in {1..3}
do
    python main.py base_simulation -sim_id $i -controller_id 1
done
