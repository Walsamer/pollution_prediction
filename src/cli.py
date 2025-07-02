import argparse
import os
import yaml
import logging
import pandas as pd

from src.preprocess import preprocess_data
from src.explaratory_data_analysis import plot_pm25_trend, plot_correlation, plot_histogram_pm25
from src.dataloaders import get_data_loaders
from src.model import choose_device, PM_Model
from src.train import train_model
from src.hyperparameter_search import hyperparameter_search

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Air Quality ML Pipeline CLI")
    parser.add_argument(
        "--run", 
        choices=["preprocess", "eda", "train", "hparam"],
        required=True,
        help="Which pipeline step to execute"
    )
    parser.add_argument(
        "--config", 
        default="configs/default.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    # load config
    cfg = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    if args.run == "preprocess":
        raw_zip = cfg["raw_zip"]
        station = cfg["station"]
        output_csv = cfg["processed_csv"]
        df = preprocess_data(raw_zip, station, output_csv)
        logger.info("Preprocessing complete, saved to %s", output_csv)

    elif args.run == "eda":
        processed_csv = cfg.get("processed_csv")
        reports_dir = cfg.get("reports_dir", "reports/")
        if processed_csv and os.path.exists(processed_csv):
            df = pd.read_csv(processed_csv, index_col=0, parse_dates=True)
        else:
            logger.error("Processed CSV not found at %s", processed_csv)
            return
        os.makedirs(reports_dir, exist_ok=True)
        plot_pm25_trend(df, reports_dir)
        plot_correlation(df, reports_dir)
        plot_histogram_pm25(df, reports_dir)
        logger.info("EDA plots saved to %s", reports_dir)

    elif args.run == "train":
        processed_csv = cfg.get("processed_csv")
        model_dir = cfg.get("model_dir", "models/")
        scaler_dir = cfg.get("scaler_dir", model_dir)

        df = pd.read_csv(processed_csv, index_col=0, parse_dates=True)
        
        #getting dataloaders
        train_loader, val_loader, test_loader = get_data_loaders(
            df,
            batch_size=cfg.get("batch_size", 64),
            scaler_dir=scaler_dir,
            val_split=cfg.get("val_split", 0.2),
            test_split=cfg.get("test_split", 0.1),
            random_seed=cfg.get("random_seed", 42)
        )

        device = choose_device()
        logger.info("Using device: %s", device)

        #building model
        for xb, _ in train_loader:  
            input_dim = xb.shape[1]
            break
        model = PM_Model(
            input_dim=input_dim,
            hidden_size=cfg.get("hidden_size"),
            dropout=cfg.get("dropout", 0.0)
        )
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            cfg=cfg
        )
        logger.info("Training complete. Best val loss: %.4f at epoch %d", history['best_val_loss'], history['best_epoch'])

    elif args.run == "hparam":
        processed_csv = cfg.get("processed_csv")
        df = pd.read_csv(processed_csv, index_col=0, parse_dates=True)
        
        # DataLoaders with initial batch_size (will be overwritten in HPO)
        train_loader, val_loader, test_loader = get_data_loaders(
            df,
            batch_size=cfg.get("batch_size", 64),
            scaler_dir=cfg.get("scaler_dir", cfg.get("model_dir", "models/")),
            val_split=cfg.get("val_split", 0.2),
            test_split=cfg.get("test_split", 0.1),
            random_seed=cfg.get("random_seed", 42)
        )
        for xb, _ in train_loader:  
            input_dim = xb.shape[1]
            break
        device = choose_device()
        logger.info("Starting hyperparameter search on device: %s", device)
        results = hyperparameter_search(
            model_cls=PM_Model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            cfg=cfg,
            input_dim=input_dim
        )
        logger.info("Hyperparameter search complete. Best params: %s, Best metric: %.4f", results.get('best_params'), results.get('best_metric'))

if __name__ == '__main__':
    main()
