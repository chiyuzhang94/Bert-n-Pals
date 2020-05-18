#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --job-name=mt
#SBATCH --output=mt.out
#SBATCH --account=rrg-mageed
#SBATCH --mail-user=zcy94@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
module load cuda cudnn
source ~/py3.6/bin/activate

rm -r save/*
export BERT_BASE_DIR=./bert-base-uncased
export BERT_PYTORCH_DIR=./bert-base-uncased
export GLUE_DIR=/home/chiyu94/scratch/Bert-n-Pals/data/semeval_normalized
export SAVE_DIR=./save


python run_multi_task.py \
  --seed 42 \
  --output_dir $SAVE_DIR/pals \
  --tasks 'all_class' \
  --sample 'anneal'\
  --nb_task 2 \
  --data_directory './data_directory.json' \
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
  --num_train_epochs 25.0 \
  --gradient_accumulation_steps 8
