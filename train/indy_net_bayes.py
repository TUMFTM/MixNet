"""Bayesian optimization of IndyNet."""
import os
import argparse
import json
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import names

from indy_net_train import main


def bayes(lr_rmse_log, lr_nll_log, lr_rmse_decay_rate):
    """Optimize inputted net."""
    print(
        "lr_rmse_log: {:2f} \t lr_nll_log: {:2f} \t lr_rmse_decay_rate: {:4f}".format(
            lr_rmse_log, lr_nll_log, lr_rmse_decay_rate
        )
    )
    common_args["lr_rmse"] = 10**lr_rmse_log
    common_args["lr_nll"] = 10**lr_nll_log
    common_args["lr_rmse_decay_rate"] = lr_rmse_decay_rate

    common_args["model_name"] = names.get_first_name()

    gain = main(common_args, verbose=True)

    return gain


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="train/configs/indy_net/default.json"
    )
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--init_points", type=int, default=20)
    parser.add_argument("--n_iter", type=int, default=30)
    args = parser.parse_args()

    if args.debug:
        args.init_points = 2
        args.n_iter = 3

    # Load config
    with open(args.config, "r") as f:
        common_args = json.load(f)

    # Network Arguments
    common_args["use_cuda"] = bool(common_args["gpu"])

    logger_path = "train/logs/indy_net_bayes"
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    try:
        # Linux
        common_args["model_name"] = args.config.split("/")[2].split(".")[0]
    except IndexError:
        # Windows
        common_args["model_name"] = args.config.split("\\")[2].split(".")[0]

    common_args["debug"] = args.debug
    common_args["tb_logs"] = "train/tb_logs/bayes"

    # Bounded region of parameter space
    pbounds = {
        "lr_rmse_log": (-2.7, -3.5),
        "lr_nll_log": (-2.7, -3.5),
        "lr_rmse_decay_rate": (0.95, 0.999),
    }

    optimizer = BayesianOptimization(
        f=bayes,
        pbounds=pbounds,
        random_state=1,
    )

    logger = JSONLogger(path=os.path.join(logger_path, "bayes_logs.json"))
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=args.init_points,
        n_iter=args.n_iter,
    )
