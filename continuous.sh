#!/bin/bash
source ~/.bashrc

if [[ $# -eq 0 ]]; then
    echo "Please provide a value for beta."
    exit 0
fi

beta=$1

# Create initial grid of simulations

python3 initial_grid.py 

# Generate first uncertainty value

python3 build_sim_library.py

# Get the max uncertainty

sigma=$(tail -1 error_evolution.dat | awk '{print $NF}')

# While max uncertainty above below limit,
# keep placing new simulations

limit=0.1

# Repeat placement til AB mag uncertainty
# falls below limit defined above

while $sigma -lt $limit;
do
    python3 build_sim_library.py
    sigma=$(tail -1 error_evolution.dat | awk '{print $NF}')
done
