#!/bin/bash
#SBATCH --job-name=isolationForestHpo
#SBATCH --ntasks=1
#SBATCH -t 0:05:00
# Specify your email address to be notified of progress.
# SBATCH --mail-user=youremailaddress@unimelb.edu.au
# SBATCH --mail-type=ALL

# Set the remote port and server
REMOTEHOST=34.168.75.39
REMOTEPORT=3306

# load modules and activate env
module purge
module load python/3.8.2
source ~/venvs/3.8.2/bin/activate

# Each trial in the study will be run in a separate job.
# The Optuna study_name has to be set to be able to continue an existing study.
OPTUNA_STUDY_NAME=duneesha_isolation_forest_run_1

OPTUNA_DB=mysql://optuna:Optuna#1234@REMOTEHOST:REMOTEPORT/DuneeshaDB

# Launch your script, giving it as arguments the database file and the study name
python if_main.py --optuna-db $OPTUNA_DB --optuna-study-name $OPTUNA_STUDY_NAME












# Load the environment variables for R

module load python/3.8.6

# The command to actually run the job
python postman_echo.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
