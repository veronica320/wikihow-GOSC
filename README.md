# wikihow-GOSC
This is the code repository associated with the paper in submission < Goal-Oriented Script Construction >. 

## Get started

### Environment

- `environment.yml` specifies the conda environment needed running the code. You can create the environment using it according to [this guildeline](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

Note that the yml file might be overkilling, since it also contains dependencies we used for other parts of the project. You might only need a few essential packages (e.g. transformers) to run the code in this repo.

- When installing transformers, make sure you install it from source and put it under the root directory of this repo. This is because we need the scripts under `transformers/examples/`. Also, please replace the `run_glue.py` file `transformers/examples/` by our `source/run_glue.py`. We modified it to allow the output of prediction probability scores.

### Data

Sample data can be downloaded from [here](https://gofile.io/d/Y9va9V) (Passcode:wikihowgosc), and put under `data_dir/`.

### Pretrained models

The pretrained models can be downloaded from [here](https://gofile.io/d/Yk9QII) (Passcode:wikihowgosc), and put under `output_dir/`.


## Repo Structure and File Format

- `data_dir/`: You should download `data.zip` and put its *subfolders* here.
	- `script_splits/`: The `script_en.json` file contains the wikiHow English script corpus, split into train and test. The file is a json consisting of two keys, `"train"` and `"test"`. Each split is a list of articles. 
	[TODO: Harry can add more details] 
	
	- `subtasks/`: This is the sample data for the two subtasks of the retrieval-based pipeline. You should format your custom data accordingly.
		- `step_en/`: This is the data for the step inference task. `train.tsv` is the *complete* train data, and `dev.tsv` is the *sample* dev data (Since we only train on 50 negative candidates cach article, but evaluate on all in-category negative candidates). 
		The data format is ```[Index]\t[Goal]\t[Candidate step]\t[Label]```. 
		Label=1 means the candidate step is a step of the given goal, 0 means otherwise.

		- `order_en/`: This is the data for the step ordering task. `train.tsv` is the *sample* train data, and `dev.tsv` is the *complete* dev data.
		The data format is ```[Index]\t[Goal]? [Step A]\t[Goal]? [Step B]\t[Label]``` (empirically the best design choice).
		Label=0 means Step A precedes B, 0 means otherwise.
		
- `output_dir/`: This is the output directory where models and predictions are stored. You should place the *subfolders* (but not the `models/` folder itself) of the downloaded `model.zip` under it. It should look like this:
	- `step_en_mbert/`
	- `order_en_mbert/`

- `source/`: The source code.
	- `finetune.py`: The code to finetune and evaluate a model on one subtask.
	- `eval_construction.py`: The code to construct final scripts from predictions of the two subtasks, and evaluate the entire pipeline on the GOSC task.
	- `run_glue.py`: The script that should be placed under your installed `transformers/examples/` directory.
- `transformers/`: The transformers package you are going to install from source. 
- `environment.yml`: The conda environment config file.



## Usage

### Finetuning models on subtasks

* Prepare your data according to the sample format. Put the `train.tsv` and `dev.tsv` files under `data_dir/{subtask_name}/`.

* Specify your own paths at the beginning of `finetune.py`.

* Go to `source/`, and run 

```
python finetune.py --mode train_eval --model [model_name] --max_seq_length [max_seq_length] --target [subtask_name] --t_bsize [t_bsize] --e_bsize [e_bsize] --lr [lr] --epochs [epochs] --logstep [logstep] --save_steps [savestep] --cuda [cuda]
```

Example:

```
python finetune.py --mode train_eval --model mbert --max_seq_length 64 --target step_en --t_bsize 32 --e_bsize 128 --lr 1e-5 --epochs 10 --logstep 40000 --save_steps 40000 --cuda 6
```

Details on the arguments are in `finetune.py`.

If you'd like to finetune a model from scratch, set the `--model` argument as `mbert`, `xlm-roberta`, etc. 

If you'd like to finetune pretrained models, set it as the name of the model directory under `output_dir`, e.g. `step_en_mbert`.


* The model output will be in `output_dir/{subtask_name}_{model_name}`, e.g. `output_dir/step_en_mbert`. It will contain the trained model (`pytorch_model.bin`) and its predictions on the dev set (`model_pred.csv`).

### Evaluating models on subtasks

* Prepare your data according to the sample format. Put the `dev.tsv` file under `data_dir/{subtask_name}/`.

* Specify your own paths at the beginning of `finetune.py`.

* Go to `source/`, and run 

```
python finetune.py --mode eval --model [model_name] --max_seq_length [max_seq_length] --target [subtask_name] --e_bsize [e_bsize] --cuda [cuda]
```

Example:

```
python finetune.py --mode eval --model mbert --max_seq_length 64 --target step_en --e_bsize 128 --cuda 6
```

* The model output will be in `output_dir/{subtask_name}_{model_name}`, e.g. `output_dir/step_en_mbert`. It will contain the model's predictions on the dev set (`model_pred.csv`) and the evaluation results (`eval_results.txt`; note that this isn't final evaluation results on the GOSC task).


### Generating scripts & Evaluating on the GOSC task

[TODO: Harry can add details]


## License
Distributed under the MIT License.
