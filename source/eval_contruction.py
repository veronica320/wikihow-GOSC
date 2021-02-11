"""
Evaluates script construction 
Relies on model output of Step Inference and Step Ordering
Example usage: 
$ python -u eval_contruction.py --lang th --task combined --model xlmr --print
will evaluate the Thai language one the combined output of step inference 
and step ordering, and print out the predicted scrips.
Written by Li Zhang <zharry@seas.upenn.edu>
"""

import argparse
from os.path import join
import csv
import json
from sklearn.metrics import recall_score
from sklearn.metrics import ndcg_score
from sklearn.metrics import accuracy_score
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
from scipy.stats import kendalltau
from collections import OrderedDict
import random
#random.seed(29)

parser = argparse.ArgumentParser()
parser.add_argument("--lang",
        default=None,
        type=str,
        required=False,
        help="Language.",
    )
parser.add_argument("--task",
        default='everything',
        type=str,
        required=False,
        help="Task: step|order|combined|everything.",
    )
parser.add_argument("--model",
        default='mbert',
        type=str,
        required=False,
        help="Model used.",
    )
parser.add_argument("--variation",
        default='',
        type=str,
        required=False,
        help="Variations: mtl, transfer",
    )
parser.add_argument('--chance', 
        help='Random chance baseline.',
        action='store_true'
    )
parser.add_argument('--print', 
        help='Print out predicted scripts to stdout.',
        action='store_true'
    )
args = parser.parse_args()

def accuracy(gold_steps,pred_steps):
  return len(list(set(gold_steps).intersection(set(pred_steps))))/len(gold_steps)
def recall_at_k(gold_steps,pred_steps,k_vals):
  y_true = [1 for s in gold_steps]
  y_pred = [1 if s in gold_steps else 0 for s in pred_steps]
  return [sum(y_pred[:k])/sum(y_true) for k in k_vals]
def ndcg(gold_steps,pred_steps,pred_confidence,k_vals):
  y_pred = np.asarray([pred_confidence])
  y_true = np.asarray([[1 if s in gold_steps else 0 for s in pred_steps]])
  return [ndcg_score([y_true[0][:k]], [y_pred[0][:k]]) for k in k_vals]
def kendall_tau(gold_steps,pred_steps):
  gold_ranks = list(range(len(gold_steps)))
  pred_ranks = [gold_steps.index(s) for s in pred_steps]
  return kendalltau(gold_ranks, pred_ranks)[0]

class EvaluateContruction():
  def __init__(self, model, lang, variation='', chance=False):
    self.model = model
    self.lang = lang
    if variation == '':
      self.gold_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/data/wikihow_data/multilingual/script_splits/script_{self.lang}.json'
      self.step_output_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/output_dir/multilingual/{self.lang}/step_{self.lang}_newsplit_{self.model}/model_pred.csv'
      self.order_output_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/output_dir/multilingual/{self.lang}/order_{self.lang}_newsplit_allpair_double_gs+gs_shuffle_{self.model}/model_pred.csv'
    elif variation == 'c0':
      self.gold_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/data/wikihow_data/multilingual/script_splits/script_{self.lang}.json'
      self.step_output_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/output_dir/multilingual/en/step_{self.lang}_newsplit_step_en_newsplit_{self.model}/model_pred.csv'
      self.order_output_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/output_dir/multilingual/en/order_{self.lang}_newsplit_allpair_double_gs+gs_shuffle_order_en_newsplit_allpair_double_gs+gs_shuffle_{self.model}/model_pred.csv'
    elif variation == 'mtl':
      self.gold_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/data/wikihow_data/multilingual/script_splits/script_{self.lang}.json'
      self.step_output_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/output_dir/multitask/step_{self.lang}_newsplit_order_{self.lang}_newsplit_allpair_double_gs+gs_shuffle/step_model_pred.csv'
      self.order_output_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/output_dir/multitask/step_{self.lang}_newsplit_order_{self.lang}_newsplit_allpair_double_gs+gs_shuffle/order_model_pred.csv'
    elif variation == 'transfer':
      self.gold_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/data/{self.lang}/script_split.json'
      self.step_output_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/output_dir/multilingual/en/step_en_{self.lang}_transfer_{self.model}/model_pred_idx.csv'
      self.order_output_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/output_dir/multilingual/en/order_en_{self.lang}_transfer_{self.model}/model_pred_idx.csv'
    elif variation == 'transfer-single':
      self.gold_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/data/{self.lang}/script_split_single.json'
      self.step_output_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/output_dir/multilingual/en/step_en_{self.lang}_transfer_{self.model}/model_pred_single.csv'
      self.order_output_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/output_dir/multilingual/en/order_en_{self.lang}_transfer_{self.model}/model_pred_single.csv'
    elif variation == 'kairos':
      self.gold_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/data/kairos/script_split.json'
      self.step_output_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/output_dir/multilingual/en/step_en_kairos_transfer_{self.model}/model_pred.csv'
      self.order_output_path = f'/mnt/nlpgridio3/data/lyuqing-zharry/wikihow_probing/output_dir/multilingual/en/order_en_kairos_transfer_{self.model}/model_pred.csv'
    self.variation = variation
    self.chance = chance
    # output
    self.goal_step_confidence = {}
    self.goal_steps_order = {}
    # gold
    self.goal_steps_gold = {}
    self.ordered_goal_steps_gold = {}
    self.read_gold()
    self.recall_avg = [0,0]
    self.ndcgs_avg = [0,0]
    self.taus_i_avg = 0
    self.accuracies_avg = 0
    self.taus_c_avg = 0

    if args.task == 'step':
      self.read_step_output()
      self.eval_step_inference()
    elif args.task == 'order':
      self.read_order_output()
      self.eval_step_ordering()
    elif args.task == 'combined':
      self.read_step_output()
      self.read_order_output()
      self.eval_combined()
    elif args.task == 'everything':
      self.read_step_output()
      self.read_order_output()
      self.eval_step_inference()
      self.eval_step_ordering()
      self.eval_combined()

  def read_gold(self):
    with open(self.gold_path) as fr:
      obj = json.load(fr)
      for article in obj['test']:
        title = article['title']
        if 'How to ' == title[:7]:
          title = title[7:]
        if 'sections' in article:
          self.goal_steps_gold[title] = []
          self.goal_steps_gold[title] = [item for sublist in [o['steps'] for o in article['sections']] for item in sublist]
          if article['ordered'] == 1:
            self.ordered_goal_steps_gold[title] = [item for sublist in [o['steps'] for o in article['sections']] for item in sublist]
        else:
          self.goal_steps_gold[title] = article['steps']
          if self.variation in ['transfer', 'transfer-single', 'kairos']:
            self.ordered_goal_steps_gold[title] = article['steps']
          elif article['ordered'] == 1:
            self.ordered_goal_steps_gold[title] = article['steps']
    
  def read_step_output(self):
    with open(self.step_output_path) as fr:
      reader = csv.DictReader(fr)
      if self.variation != 'transfer':
        for row in reader:
          if row['sentence1'] not in self.goal_step_confidence:
            self.goal_step_confidence[row['sentence1']] = {row['sentence2']: float(row['confidence'])}
          else:
            self.goal_step_confidence[row['sentence1']][row['sentence2']] = float(row['confidence'])
        self.goal_step_confidence = {k : sorted(v.items(), key=lambda item: item[1], reverse=True) for k,v in self.goal_step_confidence.items()} 
      else:
        neg_goal_to_pos_goals = {}
        for row in reader:
          goal = row['sentence1']
          step = row['sentence2']
          if '#' in goal: #pos goal
            neg_counterpart = goal.split(' #')[0]
            if neg_counterpart not in neg_goal_to_pos_goals:
              neg_goal_to_pos_goals[neg_counterpart] = [goal]
            elif goal not in neg_goal_to_pos_goals[neg_counterpart]:
              neg_goal_to_pos_goals[neg_counterpart].append(goal)
            if goal not in self.goal_step_confidence:
              self.goal_step_confidence[goal] = {step: float(row['confidence'])}
            else:
              self.goal_step_confidence[goal][step] = float(row['confidence'])
          else: #neg goal
            for pos_counterpart in neg_goal_to_pos_goals[goal]:
              self.goal_step_confidence[pos_counterpart][step] = float(row['confidence'])
        self.goal_step_confidence = {k : sorted(v.items(), key=lambda item: item[1], reverse=True) for k,v in self.goal_step_confidence.items()} 

  def read_order_output(self):
    with open(self.order_output_path) as fr:
      reader = csv.DictReader(fr)
      self.goal_steps_order = {}
      for row in reader:
        goal = row['sentence1'].split('? ')[0]
        step1 = row['sentence1'].split('? ')[1]
        step2 = row['sentence2'].split('? ')[1]
        if goal not in self.goal_steps_order:
          self.goal_steps_order[goal] = {}
        winner = [step1,step2][int(row['pred'])]
        loser = [step1,step2][1 - int(row['pred'])]
        if winner not in self.goal_steps_order[goal]: 
          self.goal_steps_order[goal][winner] = 1
        else:
          self.goal_steps_order[goal][winner] += 1
        if loser not in self.goal_steps_order[goal]: 
          self.goal_steps_order[goal][loser] = 0
      self.goal_steps_order = {k:list(v.items()) for k,v in self.goal_steps_order.items()}
      for goal in self.goal_steps_order:
        random.shuffle(self.goal_steps_order[goal])
      self.goal_steps_order = {k : sorted(v, key=lambda item: item[1], reverse=True) for k,v in self.goal_steps_order.items()}

  def eval_combined(self):
    accuracies = []
    taus = []
    correct_lengths = []
    to_print_acc = []
    for goal, gold_steps in self.goal_steps_gold.items():
      if 'How to ' == goal[:7]:
        goal = goal[7:]
      try:
        pred_steps = [t[0] for t in self.goal_step_confidence[goal]]
      except KeyError:
        continue
      if self.chance:
        random.shuffle(pred_steps)
      final_pred_steps = pred_steps[:len(gold_steps)]
      final_gold_steps = [s for s in gold_steps if s in final_pred_steps]
      acc = accuracy(gold_steps,final_pred_steps)
      accuracies.append(acc)
      if goal in self.ordered_goal_steps_gold:
        try:
          pred_ordered_steps = [t[0] for t in self.goal_steps_order[goal]]
        except KeyError:
          continue
        final_pred_ordered_steps = [s for s in pred_ordered_steps if s in final_pred_steps]
        if self.chance:
          final_pred_ordered_steps = final_gold_steps.copy()
          random.shuffle(final_pred_ordered_steps)
        try:
          tau = kendall_tau(final_gold_steps,final_pred_ordered_steps)
        except ValueError: # there're duplicate steps in gold
          try:
            tau = kendall_tau(list(OrderedDict.fromkeys(final_gold_steps)),list(OrderedDict.fromkeys(final_pred_ordered_steps)))
          except ValueError:
            print('Tau problem')
            continue
        if np.isnan(tau):
          continue
        taus.append(tau)
        if len(final_pred_steps) <= 20:
          to_print_acc.append((goal + '\n-------START-------\n' + '\n'.join(final_pred_steps) + '\n--------END------\n', acc, tau))
    self.accuracies_avg = np.mean(accuracies)
    self.taus_c_avg = np.mean(taus)
    if args.print:
      to_print_acc = sorted(to_print_acc, key=lambda item: item[1], reverse=True)
      for t in to_print_acc[:50]:
        print(t[1])
        print(t[2])
        print(t[0])

  def eval_step_inference(self):
    recalls = []
    ndcgs = []
    for goal, gold_steps in self.goal_steps_gold.items():
      if 'How to ' == goal[:7]:
        goal = goal[7:]
      try:
        pred_steps = [t[0] for t in self.goal_step_confidence[goal]]
      except KeyError:
        continue
      pred_confidence = [float(t[1]) for t in self.goal_step_confidence[goal]]
      rec = recall_at_k(gold_steps,pred_steps,[25,50])
      recalls.append(rec)
      ndcgs.append(ndcg(gold_steps,pred_steps,pred_confidence,[25,50]))
    recalls = np.array(recalls)
    self.recall_avg = np.mean(recalls,axis=0)
    self.ndcgs_avg = np.mean(ndcgs,axis=0)

  def eval_step_ordering(self):
    taus = []
    for goal, gold_steps in self.ordered_goal_steps_gold.items():
      if 'How to ' == goal[:7]:
        goal = goal[7:]
      try:
        pred_steps = [t[0] for t in self.goal_steps_order[goal]]
      except KeyError:
        continue
      try:
        tau = kendall_tau(gold_steps,pred_steps)
      except ValueError: # there're duplicate steps in gold
        try:
          tau = kendall_tau(list(OrderedDict.fromkeys(gold_steps)),list(OrderedDict.fromkeys(pred_steps)))
        except ValueError:
          #print('Tau problem')
          continue
      if np.isnan(tau):
        continue
      taus.append(tau)

    self.taus_i_avg = np.mean(taus)

  def get_results(self):
    return [self.recall_avg[0],self.recall_avg[1],self.ndcgs_avg[0],self.ndcgs_avg[1],\
    self.taus_i_avg,self.accuracies_avg,self.taus_c_avg]

def print_results(lang,a,b=None):
  if b is None:
    print(f'{lang} & {a[0]:0.3f} & {a[1]:0.3f} & {a[2]:0.3f} & {a[3]:0.3f} & {a[4]:0.3f} & {a[5]:0.3f} & {a[6]:0.3f}\\\\')
  else:
    print(f'{lang} & {a[0]:0.3f}/{b[0]:0.3f} & {a[1]:0.3f}/{b[1]:0.3f} & {a[2]:0.3f}/{b[2]:0.3f} & {a[3]:0.3f}/{b[3]:0.3f} & {a[4]:0.3f}/{b[4]:0.3f} & {a[5]:0.3f}/{b[5]:0.3f} & {a[6]:0.3f}/{b[6]:0.3f} \\\\')

if args.lang == 'all':
  langs = ["pt", "de", "fr", "ru", "it", "id", "zh", "nl", "ar", "vn", "th", "jp", "ko", "cz", "hi", "tr"]
else:
  langs = [args.lang]
for lang in langs:
  ec = EvaluateContruction(args.model, lang, args.variation, args.chance)
  results = ec.get_results()
  print_results(lang, results)
