dl_best_hparams = [
    {
        "model": "AdaCare",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 32,
        "output_dim": 1,
    },
    {
        "model": "Agent",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 32,
        "output_dim": 1,
    },
    {
        "model": "GRASP",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 128,
        "output_dim": 1,
    },
    {
        "model": "GRU",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 32,
        "output_dim": 1,
    },
    {
        "model": "LSTM",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 128,
        "output_dim": 1,
    },
    {
        "model": "MLP",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 32,
        "output_dim": 1,
    },
    {
        "model": "RETAIN",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 64,
        "output_dim": 1,
    },
    {
        "model": "RNN",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 64,
        "output_dim": 1,
    },
    {
        "model": "StageNet",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 64,
        "output_dim": 1,
    },
    {
        "model": "TCN",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 64,
        "output_dim": 1,
    },
    {
        "model": "Transformer",
        "dataset": "tjh",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 73,
        "hidden_dim": 64,
        "output_dim": 1,
    },
 {'model': 'AdaCare',
  'dataset': 'tjh',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.01,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1},
 {'model': 'Agent',
  'dataset': 'tjh',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.01,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1},
#  {'model': 'ConCare',
#   'dataset': 'tjh',
#   'task': 'los',
#   'epochs': 100,
#   'patience': 10,
#   'batch_size': 64,
#   'learning_rate': 0.001,
#   'main_metric': 'mae',
#   'demo_dim': 2,
#   'lab_dim': 73,
#   'hidden_dim': 32,
#   'output_dim': 1},
 {'model': 'GRASP',
  'dataset': 'tjh',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.01,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1},
 {'model': 'GRU',
  'dataset': 'tjh',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.01,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 128,
  'output_dim': 1},
 {'model': 'LSTM',
  'dataset': 'tjh',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.01,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 128,
  'output_dim': 1},
 {'model': 'MLP',
  'dataset': 'tjh',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.001,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 128,
  'output_dim': 1},
 {'model': 'RETAIN',
  'dataset': 'tjh',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.001,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1},
 {'model': 'RNN',
  'dataset': 'tjh',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.01,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1},
 {'model': 'StageNet',
  'dataset': 'tjh',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.01,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 128,
  'output_dim': 1},
 {'model': 'TCN',
  'dataset': 'tjh',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.01,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1},
 {'model': 'Transformer',
  'dataset': 'tjh',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.001,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 32,
  'output_dim': 1},
 {'model': 'AdaCare',
  'dataset': 'tjh',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1},
 {'model': 'Agent',
  'dataset': 'tjh',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1},
#  {'model': 'ConCare',
#   'dataset': 'tjh',
#   'task': 'multitask',
#   'epochs': 100,
#   'patience': 10,
#   'batch_size': 64,
#   'learning_rate': 0.01,
#   'main_metric': 'auprc',
#   'demo_dim': 2,
#   'lab_dim': 73,
#   'hidden_dim': 32,
#   'output_dim': 1},
 {'model': 'GRASP',
  'dataset': 'tjh',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 32,
  'output_dim': 1},
 {'model': 'GRU',
  'dataset': 'tjh',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.0001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 32,
  'output_dim': 1},
 {'model': 'LSTM',
  'dataset': 'tjh',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 128,
  'output_dim': 1},
 {'model': 'MLP',
  'dataset': 'tjh',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 32,
  'output_dim': 1},
 {'model': 'RETAIN',
  'dataset': 'tjh',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1},
 {'model': 'RNN',
  'dataset': 'tjh',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 64,
  'output_dim': 1},
 {'model': 'StageNet',
  'dataset': 'tjh',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.0001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 32,
  'output_dim': 1},
 {'model': 'TCN',
  'dataset': 'tjh',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 128,
  'output_dim': 1},
 {'model': 'Transformer',
  'dataset': 'tjh',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 64,
  'learning_rate': 0.001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 73,
  'hidden_dim': 32,
  'output_dim': 1},
    {
        "model": "AdaCare",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 32,
        "output_dim": 1,
    },
    {
        "model": "Agent",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 64,
        "output_dim": 1,
    },
    {
        "model": "GRASP",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 32,
        "output_dim": 1,
    },
    {
        "model": "GRU",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 64,
        "output_dim": 1,
    },
    {
        "model": "LSTM",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 64,
        "output_dim": 1,
    },
    {
        "model": "MLP",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 32,
        "output_dim": 1,
    },
    {
        "model": "RETAIN",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 64,
        "output_dim": 1,
    },
    {
        "model": "RNN",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 128,
        "output_dim": 1,
    },
    {
        "model": "StageNet",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.001,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 32,
        "output_dim": 1,
    },
    {
        "model": "TCN",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 128,
        "output_dim": 1,
    },
    {
        "model": "Transformer",
        "dataset": "cdsl",
        "task": "outcome",
        "epochs": 100,
        "patience": 10,
        "batch_size": 128,
        "learning_rate": 0.01,
        "main_metric": "auprc",
        "demo_dim": 2,
        "lab_dim": 97,
        "hidden_dim": 32,
        "output_dim": 1,
    },
 {'model': 'AdaCare',
  'dataset': 'cdsl',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.001,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 128,
  'output_dim': 1},
 {'model': 'Agent',
  'dataset': 'cdsl',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.01,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 32,
  'output_dim': 1},
 {'model': 'GRASP',
  'dataset': 'cdsl',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.001,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 32,
  'output_dim': 1},
 {'model': 'GRU',
  'dataset': 'cdsl',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.01,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 64,
  'output_dim': 1},
 {'model': 'LSTM',
  'dataset': 'cdsl',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.0001,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 32,
  'output_dim': 1},
 {'model': 'MLP',
  'dataset': 'cdsl',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.001,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 64,
  'output_dim': 1},
 {'model': 'RETAIN',
  'dataset': 'cdsl',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.001,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 128,
  'output_dim': 1},
 {'model': 'RNN',
  'dataset': 'cdsl',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.01,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 128,
  'output_dim': 1},
 {'model': 'StageNet',
  'dataset': 'cdsl',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.001,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 128,
  'output_dim': 1},
 {'model': 'TCN',
  'dataset': 'cdsl',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.001,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 128,
  'output_dim': 1},
 {'model': 'Transformer',
  'dataset': 'cdsl',
  'task': 'los',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.01,
  'main_metric': 'mae',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 128,
  'output_dim': 1},
 {'model': 'AdaCare',
  'dataset': 'cdsl',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 64,
  'output_dim': 1},
 {'model': 'Agent',
  'dataset': 'cdsl',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 32,
  'output_dim': 1},
 {'model': 'GRASP',
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
  'output_dim': 1},
 {'model': 'GRU',
  'dataset': 'cdsl',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 32,
  'output_dim': 1},
 {'model': 'LSTM',
  'dataset': 'cdsl',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 32,
  'output_dim': 1},
 {'model': 'MLP',
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
  'output_dim': 1},
 {'model': 'RETAIN',
  'dataset': 'cdsl',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 128,
  'output_dim': 1},
 {'model': 'RNN',
  'dataset': 'cdsl',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 32,
  'output_dim': 1},
 {'model': 'StageNet',
  'dataset': 'cdsl',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.0001,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 32,
  'output_dim': 1},
 {'model': 'TCN',
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
  'output_dim': 1},
 {'model': 'Transformer',
  'dataset': 'cdsl',
  'task': 'multitask',
  'epochs': 100,
  'patience': 10,
  'batch_size': 128,
  'learning_rate': 0.01,
  'main_metric': 'auprc',
  'demo_dim': 2,
  'lab_dim': 97,
  'hidden_dim': 128,
  'output_dim': 1}]