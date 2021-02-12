#!/usr/bin/env python
# coding: utf-8
import transformers
import tensorboardX
import mxnet
import os 
import sys
import argparse

# root_dir = '/path/to/wikihow-GOSC' # specify your own
# os.environ['PYTORCH_TRANSFORMERS_CACHE']="/path/to/.cache/transformers" # specify your own

# Example:
root_dir = '/nlp/data/lyuqing-zharry/wikihow-GOSC'
os.environ['PYTORCH_TRANSFORMERS_CACHE']= "/nlp/data/lyuqing-zharry/wikihow_probing/.cache/transformers"

os.chdir(root_dir)

# set max cpu threads
os.environ['export OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OPENMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'


parser = argparse.ArgumentParser(description='Process finetune config.')
parser.add_argument("--target",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .csv files for the task.",
    )
parser.add_argument("--mode",
        default='train_eval',
        type=str,
        required=False,
        help="What you want to do. Can be train_eval | eval.",
    )
parser.add_argument("--cuda",
        default=None,
        type=str,
        required=True,
        help="The GPU indices to use.",
    )
parser.add_argument("--model",
        default='roberta',
        type=str,
        required=False,
        help="Model architecture to use.",
    )
parser.add_argument(
    "--t_bsize", 
    default='8', 
    type=str,
    help="Batch size per GPU/CPU for training."
)
parser.add_argument(
    "--e_bsize", 
    default='8', 
    type=str,
    help="Batch size per GPU/CPU for evaluation."
)
parser.add_argument(
    "--max_seq_length",
    default='80',
    type=str,
    help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
    "--lr", 
    default='5e-5', 
    type=str,
    help="The initial learning rate for Adam.")
parser.add_argument(
    "--epochs", 
    default='3', 
    type=str,
    help="Total number of training epochs to perform."
)
parser.add_argument(
    "--logstep",
    default='500', 
    type=str,
    help="Log every X steps.",
)
parser.add_argument(
    "--save_steps", 
    default='-1', 
    type=str, 
    help="Save steps.")


args = parser.parse_args()

# model abbr to model type 
model_type_dict = {
  'roberta': 'roberta',
  'roberta-l': 'roberta',
  'bert': 'bert',
  'mbert': 'bert',
  'xlmr': 'xlm-roberta',
  'xlmr-l': 'xlm-roberta',
  'order_en_mbert': ' bert', # our own model
  'step_en_mbert': 'bert', # our own model
}

# model abbr to model name
model_name_dict = {
  'roberta': 'roberta-base',
  'roberta-l': 'roberta-large',
  'bert': 'bert-base-uncased',
  'mbert': 'bert-base-multilingual-uncased',
  'xlmr': 'xlm-roberta-base',
  'xlmr-l': 'xlm-roberta-large',
}

os.environ['MODEL_TYPE'] = model_type_dict[args.model]

if args.model in model_name_dict.keys():
	os.environ['MODEL_NAME_OR_PATH'] = model_name_dict[args.model] # existing ckpts
else:
	os.environ['MODEL_NAME_OR_PATH'] = f'output_dir/{args.model}' # our own model


os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

# sequence classification task
os.environ['TASK_NAME'] = 'WNLI'
os.environ['SCRIPT_NAME'] = 'run_glue.py'


os.environ['DATA_DIR'] = f'data_dir/subtasks/{args.target}'
os.environ['OUTPUT_DIR'] = f'output_dir/{args.target}_{args.model}'


print('Running mode: ', args.mode)


## train and eval
if args.mode == 'train_eval':
    os.system(f'python -u transformers/examples/$SCRIPT_NAME \
                --model_type $MODEL_TYPE \
                --task_name $TASK_NAME \
                --model_name_or_path $MODEL_NAME_OR_PATH \
                --data_dir $DATA_DIR \
                --output_dir $OUTPUT_DIR \
                --do_train \
                --do_eval \
                --do_lower_case \
                --save_steps {args.save_steps} \
                --evaluate_during_training \
                --learning_rate {args.lr} \
                --num_train_epochs {args.epochs} \
                --max_seq_length {args.max_seq_length} \
                --per_gpu_eval_batch_size={args.e_bsize} \
                --per_gpu_train_batch_size={args.t_bsize} \
                --gradient_accumulation_steps 2 \
                --overwrite_output \
                --fp16 \
                --prob \
                --logging_steps {args.logstep}'
             )
    
## eval only
elif args.mode == 'eval':
	os.system(f'python -u transformers/examples/$SCRIPT_NAME \
            --model_type $MODEL_TYPE \
            --task_name $TASK_NAME \
            --model_name_or_path $MODEL_NAME_OR_PATH \
            --do_eval \
            --do_lower_case \
            --data_dir $DATA_DIR \
            --output_dir $MODEL_NAME_OR_PATH \
            --max_seq_length {args.max_seq_length} \
            --per_gpu_eval_batch_size={args.e_bsize} \
            --overwrite_output \
            --fp16 \
            --prob'
         )

