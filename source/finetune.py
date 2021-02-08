#!/usr/bin/env python
# coding: utf-8
import transformers
import tensorboardX
import mxnet
import os 
import sys
import argparse

# root_dir = '/path/to/wikihow-GOSC' // specify your own
# os.environ['PYTORCH_TRANSFORMERS_CACHE']="/path/to/.cache/transformers" // specify your own
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
    "--no_fp16",
    action="store_true",
    help="Not to use 16-bit precision training.",
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

model_name = {
  'roberta': 'roberta-base',
  'roberta-l': 'roberta-large',
  'bert': 'bert-base-uncased',
  'mbert': 'bert-base-multilingual-uncased',
  'xlnet': 'xlnet-base-cased',
  'xlmr': 'xlm-roberta-base',
  'xlmr-l': 'xlm-roberta-large',
  'gpt': 'gpt2',
}

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

# sequence classification task
os.environ['TASK_NAME'] = 'WNLI'
os.environ['SCRIPT_NAME'] = 'run_glue.py'

# TODO
os.environ['DATA_DIR'] = f'data/{args.target.split('_')[1]}'
os.environ['OUTPUT_DIR'] = f'output_dir/{args.target.split('_')[1]}'
  
os.environ['DATA_CONFIG_STR'] = args.target 

os.environ['OUTPUT_CONFIG_STR'] = args.target + '_' + args.model

print('Running mode: ', args.mode)


## train and eval
if args.mode == 'train_eval':
    os.system(f'python -u transformers/examples/$SCRIPT_NAME \
                --model_type {args.model} \
                --task_name $TASK_NAME \
                --model_name_or_path {model_name[args.model]} \
                --do_train \
                --do_eval \
                --do_lower_case \
                --save_steps {args.save_steps} \
                --data_dir $DATA_DIR/$DATA_CONFIG_STR \
                --evaluate_during_training \
                --learning_rate {args.lr} \
                --num_train_epochs {args.epochs} \
                --max_seq_length {args.max_seq_length} \
                --output_dir $OUTPUT_DIR/$OUTPUT_CONFIG_STR \
                --per_gpu_eval_batch_size={args.e_bsize} \
                --per_gpu_train_batch_size={args.t_bsize} \
                --gradient_accumulation_steps 2 \
                --overwrite_output \
                {'--fp16' if not args.no_fp16 else ''} \
                --prob \
                --logging_steps {args.logging_steps}'
             )
    
## eval only
elif args.mode == 'eval':
    if args.source: # evaluate a trained model
        os.system('python -u transformers/examples/$SCRIPT_NAME \
                    --model_type '+ args.model + \
                  ' --task_name $TASK_NAME \
                    --model_name_or_path $OUTPUT_CONFIG_STR \
                    --do_eval \
                    --do_lower_case \
                    --data_dir $DATA_DIR/$DATA_CONFIG_STR \
                    --max_seq_length ' + args.max_seq_length + \
                  ' --output_dir $OUTPUT_CONFIG_STR \
                    --per_gpu_eval_batch_size=' + args.e_bsize + \
                  ' --overwrite_output' + \
                 (' --fp16 ' if not args.no_fp16 else '') + \
                 (' --prob ' if args.prob else '') + \
                  ' --run_name $RUN_NAME' + \
                 (' --nowab ' if args.nowab else '') + \
                 (' --overwrite_cache ' if args.overwrite_cache else '') + \
                 (' --no_cache ' if args.no_cache else '')
                 )
    else: # evaluate an untrained model
        os.system('python -u transformers/examples/$SCRIPT_NAME \
                    --model_type '+ args.model + \
                  ' --task_name $TASK_NAME \
                    --model_name_or_path ' + model_name[args.model] + \
                  ' --do_eval \
                    --do_lower_case \
                    --data_dir $DATA_DIR/$DATA_CONFIG_STR \
                    --max_seq_length ' + args.max_seq_length + \
                  ' --output_dir $OUTPUT_DIR/$OUTPUT_CONFIG_STR \
                    --per_gpu_eval_batch_size=' + args.e_bsize + \
                  ' --overwrite_output' + \
                 (' --fp16 ' if not args.no_fp16 else '') + \
                 (' --prob ' if args.prob else '') + \
                  ' --run_name $RUN_NAME' + \
                 (' --nowab ' if args.nowab else '') + \
                 (' --overwrite_cache ' if args.overwrite_cache else '')+ \
                 (' --no_cache ' if args.no_cache else '')
                 )
