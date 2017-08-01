#!/bin/bash
#Sample command lines to run the cat_vs_dog example in Google Cloud

# This sample assumes you're already setup for using CloudML.  If this is your
# first time with the service, start here:
# https://cloud.google.com/ml/docs/how-tos/getting-set-up



# Declare some environment variables used in data preprocessing and training.

# The following variables may need to be changed based on your own config.

BUCKET_NAME=gs://mrudula_yt8m_train_bucket
# (One Time) Create a storage bucket to store training logs and checkpoints.
gsutil mb -l us-east1 $BUCKET_NAME
# Submit the training job.
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.task \
--runtime-version 1.2 \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu-distributed.yaml \
-- --train_data_pattern='gs://youtube8m-ml-us-east1/1/video_level/train/train*.tfrecord' \
--model=MoeModel \
--train_dir=$BUCKET_NAME/yt8m_train_video_level_moe_model_experiment_task