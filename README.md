# wikihow-GOSC
This is the code repository accompanying the paper to appear at [INLG2021](https://inlg2021.github.io/): [< Goal-Oriented Script Construction >](https://arxiv.org/abs/2107.13189). 

## Get started

### Environment

- `environment.yml` specifies the conda environment needed running the code. You can create the environment using it according to [this guildeline](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

- When installing transformers, make sure you [**install it from source**](https://github.com/huggingface/transformers/tree/v2.4.1#from-source) and put it under the root directory of this repo. This is because we need the scripts under `transformers/examples/`. Also, please replace the `run_glue.py` file `transformers/examples/` by our `source/run_glue.py`. We modified it to allow the output of prediction probability scores.

### Data

#### Multilingual wikiHow Script Corpus
The entire corpus is available [here](https://drive.google.com/file/d/1AqAocrNFEPhBAfa5ATCj-3xMWbq659ME/view?usp=sharing).

#### Sample data to run experiments
To run the experiments following the instructions in the repo, you can use the sample data [here](https://drive.google.com/file/d/1MfoI2TfMgKj3lxnto2rMFblSYVVBUPH0/view?usp=sharing), and put it under `data_dir/`.

### Pretrained models

The pretrained models can be downloaded from [here](https://drive.google.com/file/d/1J9Vnrh1tBRnOrLnqJT2prGqm3v-6EqO1/view?usp=sharing), and put under `output_dir/`.


## Repo Structure and File Format

- `data_dir/`: You should download `data.zip` and put its *subfolders* here.
	- `script_splits/`: The `script_en.json` file contains an English sample of the wikiHow English script corpus, split into train and test. The file is a json consisting of two keys, `"train"` and `"test"`. Each split is a list of articles. See more details in the accompanying README.
	
	- `subtasks/`: This is the **sample data** for the two subtasks of the retrieval-based pipeline. `train.tsv` contains the training data, and `dev.tsv` contains the evaluation data. Note that this is for demonstration purposes only. If you want to reproduce the results in our paper, you need to download the full corpus above ("Multilingual wikiHow Script Corpus"). If you only want to construct custom scripts using our pretrained models, you can refer to the dev files and format your data accordingly. The dev files are for one example target script, "Dress Effectively".
		- `step_en/`: This is the data for the Step Inference task.
		The data format is ```[Index]\t[Goal]\t[Candidate step]\t[Label]```. 
		Label=1 means the candidate step is a step of the given goal, 0 means otherwise.

		- `order_en/`: This is the data for the Step Ordering task.  
		The data format is ```[Index]\t[Goal]? [Step A]\t[Goal]? [Step B]\t[Label]``` (empirically the best design choice).
		Label=0 means Step A precedes B, 0 means otherwise.
		Note that the step candidates in our sample `dev.tsv` are **gold** steps, so that the Step Ordering module can be evaluated independently. If you want to run the entire retrieval-based pipeline, then you should take the output of the Step Inference task and format the top L (=script length) retrieved step candidates in this way, as input to the Ordering module.
		
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

### Generation-based pipeline

Please follow instructions in [this colab notebook](https://colab.research.google.com/drive/1W_-RcZD-A2SfkqOkOOI0mNJEy9f5J9Uz?usp=sharing).

### Retrieval-based pipeline

If you want to finetune the pipeline yourself, please start from step A. If you want to directly do inference with our pretrained pipeline, please start from step B.

#### A. Finetuning & evaluating models on subtasks

* Prepare your data according to the sample format (See `Repo Structure and File Format` -> `subtasks/`). Put the `train.tsv` and `dev.tsv` files under `data_dir/{subtask_name}/`.

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

If you'd like to finetune pretrained models, set it as the name of the model directory _under_ `output_dir`, e.g. `step_en_mbert`. Note that you shouldn't include `output_dir/` in the argument.


* The model output will be in `output_dir/{subtask_name}_{model_name}`, e.g. `output_dir/step_en_mbert`. It will contain the trained model (`pytorch_model.bin`) and its predictions on the dev set (`model_pred.csv`).

#### B. Doing inference & Evaluating on two subtasks

If you only want to evaluate models on the two subtasks (Step Inference, Step Ordering) independently, then you can do the following steps for both in parallel. If you want to use the entire retrieval-based pipeline to construct scripts, then you should do the following steps for Step Inference first, and then use its output as the input to the Step Ordering subtask.

* If you haven't done A, prepare your evaluation data according to the sample format (See `Repo Structure and File Format` -> `subtasks/`). Put the `dev.tsv` file under `data_dir/{subtask_name}/`.

* Put the model you want to evaluate under `output_dir/{model_name}`. If you started from A, the models should already be there. Otherwise, you should download our pretrained models under `Get started`, and put them under `output_dir`.

* Specify your own paths at the beginning of `finetune.py`.

* Go to `source/`, and run 

```
python finetune.py --mode eval --model [model_name] --max_seq_length [max_seq_length] --target [subtask_name] --e_bsize [e_bsize] --cuda [cuda]
```

Example:

```
python finetune.py --mode eval --model step_en_mbert --max_seq_length 64 --target step_en --e_bsize 128 --cuda 6
```

* The model output will be in `output_dir/{model_name}`, i.e. the model directory you have initially, e.g. `output_dir/step_en_mbert`. It will contain the model's predictions on the dev set (`model_pred.csv`) and the evaluation results (`eval_results.txt`; note that this isn't final evaluation results on the GOSC task).


#### C. Generating scripts & Evaluating on the GOSC task

Using the model output from B, evaluation and generation can be done with

```
python eval_contruction.py --lang [language] --model [model_name] --task [step|order|combined|everything] (optional) --print
```
If `--task` is set to _step_, recall and gain are measured; if _order_, Kendall's Tau by ordering the gold script is measured; if _combined_, recall, gain and Tau of the generated script is measured; if _everything_, all of the above are measured. If `--print` is specified, the constructed scritp will be directed to standard output. 
Example:
```
python eval_contruction.py --lang en --model mbert --task combined --print
```


## License
Distributed under the MIT License.
