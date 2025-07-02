import os
import logging
import itertools
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Any, Dict
from src.train import train_model

logger = logging.getLogger(__name__)

def hyperparameter_search(model_cls: Any, train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader, device: torch.device, cfg: Dict[str, Any], input_dim: int
) -> Dict[str, Any]:
    """
    Perform grid or random hyperparameter search based on the provided configuration.

    Parameters:
        model_cls : Any
            Model class to instantiate for each trial (must accept kwargs from grid).
        train_loader : DataLoader
            DataLoader for training data.
        val_loader : DataLoader
            DataLoader for validation data.
        device : torch.device
            Device on which to train (cpu, cuda, etc.).
        cfg : dict
            Full configuration dict (parsed from YAML). Must contain a 'hyperparameter_search' section.

    Returns:
        Dict[str, Any]
            Dictionary with keys:
            - 'best_params': dict of best hyperparameters
            - 'best_metric': float best metric value
    """
    hcfg = cfg.get('hyperparameter_search', {})
    if not hcfg.get('enabled', False):
        logger.info("Hyperparameter search disabled in config.")
        return {}

    param_grid: Dict[str, list] = hcfg.get('param_grid', {})
    strategy = hcfg.get('strategy', 'grid')
    metric_name = hcfg.get('metric', 'val_loss')
    max_trials = hcfg.get('max_trials')
    output_dir = cfg.get('model_dir', 'models/')
    os.makedirs(output_dir, exist_ok=True)

    #Generating trial combinations
    keys = list(param_grid.keys())
    all_combos = list(itertools.product(*(param_grid[k] for k in keys)))
    if max_trials:
        all_combos = all_combos[:max_trials]

    results = []
    best_metric = float('inf')
    best_params = {}

    for idx, combo in enumerate(all_combos, start=1):
        params = dict(zip(keys, combo))
        logger.info("Trial %d/%d with params: %s", idx, len(all_combos), params)

        #Instantiating model
        model = model_cls(
            input_dim=input_dim,
            hidden_size=params.get('hidden_size'),
            dropout=params.get('dropout')
        ).to(device)

        # Updating and training modeltrial
        trial_cfg = cfg.copy()
        trial_cfg.update(params)

        history = train_model( 
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            cfg=trial_cfg
        )

        metric_val = history.get(metric_name)
        if metric_val == 'val_loss':
            metric_val = history['best_val_loss']
        else:
            raw = history.get(metric_name)
            metric_val = raw[-1]
        results.append({**params, metric_name: metric_val})

        if metric_val < best_metric: #comparing models and keeping the best
            best_metric = metric_val
            best_params = params
            path = os.path.join(output_dir, 'best_hpo_model.pth')
            torch.save(model.state_dict(), path)
            logger.info("New best model saved to %s with %s=%.4f", path, metric_name, best_metric)

    #saving the results. ('hpo_results.csv')
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'hpo_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info("Saved hyperparameter search results to %s", csv_path)

    plt.figure()    #plot metrics ('hpo_results_plot.png')
    plt.plot(range(1, len(results)+1), [r[metric_name] for r in results], marker='o')
    plt.xlabel('Trial')
    plt.ylabel(metric_name)
    plt.xticks(range(1, len(results)+1))
    plt.title('Hyperparameter Search {}'.format(metric_name))
    plot_path = os.path.join(output_dir, 'hpo_results_plot.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    logger.info("Saved hyperparameter search plot to %s", plot_path)

    return {'best_params': best_params, 'best_metric': best_metric}
