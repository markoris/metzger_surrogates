#!/bin/bash
source ~/.bashrc

if [[ $# -eq 0 ]]; then
    echo "Please provide a value for beta."
    exit 0
fi

beta=$1

# Create initial grid of simulations

python3 -u initial_grid.py $beta 

# Generate first uncertainty value

python3 -u build_sim_library.py $beta

# Get the max uncertainty

sigma=$(tail -1 error_evolution_beta${beta}.dat | awk '{print $NF}')

# While max uncertainty above below limit,
# keep placing new simulations

limit=0.1

# Repeat placement til AB mag uncertainty
# falls below limit defined above

while [ "$(echo "${sigma} < ${limit}" | bc)" ];
do
    python3 -u build_sim_library.py $beta
    sigma=$(tail -1 error_evolution_beta${beta}.dat | awk '{print $NF}')
done
