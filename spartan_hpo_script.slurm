#!/bin/bash
#SBATCH --job-name=isolationForestHpoArray
#SBATCH -t 0:15:00
#SBATCH --array=1-300%3
#SBATCH --gres=gpu:1
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --mem=8G

# Set the remote port and server
REMOTEHOST=34.105.9.156
REMOTEPORT=3306

# load modules and activate env
module purge
module load python/3.8.2
source ~/venvs/3.8.2/bin/activate

# Each trial in the study will be run in a separate job.
# The Optuna study_name has to be set to be able to continue an existing study.
OPTUNA_STUDY_NAME=duneesha_iforest_run_1

OPTUNA_DB=mysql://optuna:Optuna#1234@$REMOTEHOST:$REMOTEPORT/DuneeshaDB

# Launch your script, giving it as arguments the database file and the study name
python if_main.py --optuna-db $OPTUNA_DB --optuna-study-name $OPTUNA_STUDY_NAME

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
