#!/usr/bin/env python
# coding: utf-8

# # Plot Scripts

# In[ ]:


import numpy as np
import pandas as pd
from pathlib import Path

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransfor
from matplotlib.ticker import FuncFormatter
import seaborn as sns

plt.style.use('default')
plt.rcParams['axes.facecolor']='white'
plt.rcParams.update({"axes.grid" : True, "grid.color": "gainsboro"})
plt.rcParams['legend.frameon']=True
plt.rcParams['legend.facecolor']='white'
plt.rcParams['legend.edgecolor']='grey'
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"]  = 1
Path("logs/figures").mkdir(parents=True, exist_ok=True)

# ## Draw OSMAE / ES scores on different thresholds

# In[ ]:


from datasets.loader.load_los_info import get_los_info
from datasets.loader.datamodule import EhrDataModule
from pipelines import DlPipeline
import lightning as L

# In[ ]:


# init config (CDSL dataset, TCN multitask model, fold-0, seed-0)
config = {
  'model': 'TCN',
  'dataset': 'cdsl',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 128,
  'output_dim': 1,
  }

thresholds = np.arange(0,10,0.1)[1:].tolist()

# load CDSL fold-0 data
los_config = get_los_info(f'datasets/{config["dataset"]}/processed/fold_0')
los_config['threshold'] = thresholds
config.update({"los_info": los_config})
dm = EhrDataModule(f'datasets/{config["dataset"]}/processed/fold_0', batch_size=config["batch_size"])

# load TCN multitask model
checkpoint_path = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold0-seed0/checkpoints/best.ckpt'
pipeline = DlPipeline(config)
trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=False, num_sanity_val_steps=0)
trainer.test(pipeline, dm, ckpt_path=checkpoint_path)

# get scores
perf = pipeline.test_performance

# In[ ]:


print(len(perf['osmae_list']), len(perf['es_list']))
es = perf['es_list'][::4]
osmae = perf['osmae_list'][::4]
thres = thresholds[::4]
print(len(es), len(osmae), len(thres))

# In[ ]:


# ES Score
ax = sns.regplot(x=thres, y=es, marker="o", color="g", line_kws={"color": "grey", "linestyle": "-", "linewidth": "1"}, ci=99.9999)
plt.xlabel('Threshold γ')
plt.ylabel('ES Score')

plt.savefig('logs/figures/es_trend.pdf', dpi=500, format="pdf", bbox_inches="tight")
plt.show()

# In[ ]:


# OSMAE Score
ax = sns.regplot(x=thres, y=osmae, marker="o", color="dodgerblue", line_kws={"color": "grey", "linestyle": "-", "linewidth": "1"}, ci=99.9999)
plt.xlabel('Threshold γ')
plt.ylabel('OSMAE Score')

plt.savefig('logs/figures/osmae_trend.pdf', dpi=500, format="pdf", bbox_inches="tight")
plt.show()

# ## Draw feature embedding
# 
# compare multi-task and two-stage setting

# In[ ]:


import lightning as L
import torch
from sklearn.manifold import TSNE

from datasets.loader.load_los_info import get_los_info
from datasets.loader.datamodule import EhrDataModule
from pipelines import DlPipeline

# ### CDSL dataset, TCN multitask model, fold-0, seed-0

# In[ ]:


# init config
config = {
  'model': 'TCN',
  'dataset': 'cdsl',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 81920,
  'learning_rate': 0.001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 128,
  'output_dim': 1,
  }

# load CDSL fold-0 data
los_config = get_los_info(f'datasets/{config["dataset"]}/processed/fold_0')
config.update({"los_info": los_config})
dm = EhrDataModule(f'datasets/{config["dataset"]}/processed/fold_0', batch_size=config["batch_size"])

# load TCN multitask model
checkpoint_path = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold0-seed0/checkpoints/best.ckpt'
pipeline = DlPipeline(config)
trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=False, num_sanity_val_steps=0)
trainer.test(pipeline, dm, ckpt_path=checkpoint_path)

# get scores
embedding = pipeline.embedding

# In[ ]:


# fetch each patient's last visit's embedding
lens = pipeline.test_outputs['lens']
outcomes = []
flattened_outcomes = pipeline.test_outputs['labels'][:,0]
idx=0
for i in range(len(lens)):
    outcomes.append(flattened_outcomes[idx])
    idx+=lens[i]

# Initialize an empty list to store selected tensors
selected_tensors = []
# Iterate over ts_array
for i in range(len(lens)):
    l = lens[i]
    selected_tensor = embedding[i, l - 1, :]
    selected_tensors.append(selected_tensor)
# Concatenate all selected tensors along the time step dimension (axis=1)
patient_embeds = torch.stack(selected_tensors, dim=0)

print(embedding.shape, len(lens), len(outcomes), sum(lens), type(lens), lens.shape)
print(patient_embeds.shape)

outcomes = np.expand_dims(np.array(outcomes), axis=1)


# In[ ]:


projected = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(patient_embeds)
concatenated = np.concatenate([projected, outcomes], axis=1)

df = pd.DataFrame(concatenated, columns = ['Component 1', 'Component 2', 'Outcome'])
df['Outcome'] = df['Outcome'].replace({1: 'Dead', 0: 'Alive'})

sns.scatterplot(data=df, x="Component 1", y="Component 2", hue="Outcome", style="Outcome", palette=["C2", "C3"], alpha=0.5)
plt.savefig(f'logs/figures/tcn_multitask_embedding_tsne.pdf', dpi=500, format="pdf", bbox_inches="tight")
