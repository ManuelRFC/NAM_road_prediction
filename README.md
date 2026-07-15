# NAM_road_prediction
# NAM Road Prediction Project

## Overview
This project uses geospatial analysis and random forest machine learning to predict road networks. 

## Project Structure
* `requirements.txt` : The Python environment "DNA" (packages and versions).
* `README.md` : This instruction manual.
* `scripts/` : (Folder) Where all the .py Python code lives.
* `data/` : **NOTE: This folder is NOT on GitHub.** It lives on the FUCHS /work/ drive due to size limits. It contains the raw .tif satellite images.

## How to Run This Code on FUCHS-CSC
1. Log in to the cluster.
2. Activate the environment: `source ~/my_environments/geo_env/bin/activate`
3. Navigate to the project folder: `cd ~/NAM_road_prediction`
4. Submit the job: `sbatch job.sh`