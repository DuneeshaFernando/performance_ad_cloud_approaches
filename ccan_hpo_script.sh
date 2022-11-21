#!/bin/bash
#SBATCH -A def-kgroling
#SBATCH --array 1-300%3   # This will launch N jobs, but only allow M to run in parallel
#SBATCH --time 0-00:05:00     # Each of the N jobs will have the time limit defined in here.
#SBATCH --cpus-per-task=4
#SBATCH --mem=1000M

# Set the remote port and server
REMOTEHOST=34.168.75.39
REMOTEPORT=3306

# Set the local params
LOCALHOST=localhost
for ((i=0; i<10; ++i)); do
  LOCALPORT=$(shuf -i 1024-65535 -n 1)
  ssh beluga3 -L $LOCALPORT:$REMOTEHOST:$REMOTEPORT -N -f && break
done || { echo "Giving up forwarding license port after $i attempts..."; exit 1; }

# load modules and activate env
module load python
module load scipy-stack
source $HOME/mdi-optuna/bin/activate

# Each trial in the study will be run in a separate job.
# The Optuna study_name has to be set to be able to continue an existing study.
OPTUNA_STUDY_NAME=duneesha_isolation_forest_run_1

OPTUNA_DB=mysql://optuna:Optuna#1234@$LOCALHOST:$LOCALPORT/OptunaDB

# Launch your script, giving it as arguments the database file and the study name
python random_forest_hpo_script.py --optuna-db $OPTUNA_DB --optuna-study-name $OPTUNA_STUDY_NAME