#!/bin/bash

#PBS -l walltime=3:00:00,select=1:ncpus=2:ompthreads=12:ngpus=1:mem=64gb
#PBS -N multi-task
#PBS -A st-amuham01-1-gpu
#PBS -M chiyuzh@mail.ubc.ca
#PBS -m abe
#PBS -o output.txt
#PBS -e error.txt

rm -r /scratch/st-amuham01-1/chiyuzh/Bert-n-Pals/save/*
export BERT_BASE_DIR=/scratch/st-amuham01-1/chiyuzh/Bert-n-Pals/bert-base-cased
export BERT_PYTORCH_DIR=/scratch/st-amuham01-1/chiyuzh/Bert-n-Pals/bert-base-cased
export GLUE_DIR=/scratch/st-amuham01-1/chiyuzh/Bert-n-Pals/data/semeval_normalized
export SAVE_DIR=/scratch/st-amuham01-1/chiyuzh/Bert-n-Pals/save

module load python/3.6.8
module load gcc
module load cuda

echo $batch_id

python3 /scratch/st-amuham01-1/chiyuzh/Bert-n-Pals/run_multi_task.py \
  --seed 42 \
  --output_dir $SAVE_DIR/pals \
  --tasks 'all_class' \
  --epoch_step 200 \
  --sample 'anneal'\
  --nb_task 4 \
  --data_directory '/scratch/st-amuham01-1/chiyuzh/Bert-n-Pals/data_directory.json' \
  --multi \
  --do_lower_case \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/ \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/pals_config.json \
  --init_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
  --max_seq_length 64 \
  --train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 2.0 \
  --gradient_accumulation_steps 8
