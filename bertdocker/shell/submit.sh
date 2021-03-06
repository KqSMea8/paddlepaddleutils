#!/bin/bash

if [[ $# != 1 ]]; then
    echo "input: job_conf"
    exit -1
fi

job_conf=$1
source $job_conf


if [ $HGCP_CLIENR_BIN ]; then
    HGCP_CLIENR_BIN=$HGCP_CLIENR_BIN
else
    HGCP_CLIENR_BIN=~/.hgcp/software-install/HGCP_client/bin
fi

hdfs_ugi=(${ugi//,/ })
${HGCP_CLIENR_BIN}/submit \
        --submitter $submitter \
        --hdfs $hdfs_path \
        --hdfs-user ${hdfs_ugi[0]} \
        --hdfs-passwd ${hdfs_ugi[1]} \
        --hdfs-path $hdfs_output \
        --file-dir ./ \
        --job-name $task_name \
        --num-nodes $nodes \
        --queue-name nlp-temp \
        --num-task-pernode 1 \
        --gpu-pnode 8 \
        --time-limit $walltime \
        --job-script job.sh
        
#-s liyukun01 
#        --queue-name yq01-p40-3-8 \
#        --queue-name yq01-v100-box-1-8 \
#        --queue-name yq01-p40-4-8 \
        #--queue-name yq01-p40-3-8 \
