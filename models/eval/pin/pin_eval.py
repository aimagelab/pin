import argparse
import datetime
import jsonlines
import habitat
from tqdm import tqdm
import os
from collections import defaultdict
from typing import Dict, Optional

from habitat import Env

from utils.new_top_down_map import *
from utils.oracle_navigators import ShortestPathFollowerAgentPIN


def evaluate(config_env, num_episodes: Optional[int] = None) -> Dict[str, float]:

    split = config_env.habitat.dataset.split
    env = Env(config=config_env)
    agent = ShortestPathFollowerAgentPIN(env.sim, config_env)

    if num_episodes is None:
        num_episodes = len(env.episodes)
    else:
        assert num_episodes <= len(env.episodes), (
            "num_episodes({}) is larger than number of episodes "
            "in environment ({})".format(num_episodes, len(env.episodes))
        )
    assert num_episodes > 0, "num_episodes should be greater than 0"

    agg_metrics: Dict = defaultdict(float)

    os.makedirs(f"dbg_imgs/pin/{split}", exist_ok=True)
    os.makedirs(f"results/pin/{split}", exist_ok=True)
    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
    results_file = f"results/pin/{split}/{timestamp}_results.jsonl"

    count_episodes = 0
    failed_episodes = []
    with tqdm(total=num_episodes) as pbar:
        while count_episodes < num_episodes:
            observations = env.reset()
                
            agent.reset()

            pbar.update(1)
            steps = 0
            while not env.episode_over:
                
                action = agent.act(observations, env)
                observations = env.step(action)
                steps += 1

            metrics = env.get_metrics()
            pbar.set_description(
                f"{count_episodes}: length:{metrics['episode_length']}, s:{metrics['success']}, spl:{round(metrics['spl'], 2)}, cat_spl:{round(metrics['cat_spl'], 2)}")
            with jsonlines.open(results_file, mode="a") as f:
                f.write(metrics)
                
            for m, v in metrics.items():
                if isinstance(v, dict):
                    for sub_m, sub_v in v.items():
                        agg_metrics[m + "/" + str(sub_m)] += sub_v
                else:
                    agg_metrics[m] += v
            count_episodes += 1

    avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
    print(avg_metrics)

    return avg_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/models/pin/pin.yaml")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = habitat.get_config(args.config_file, args.opts)
    evaluate(config_env=config)


if __name__ == "__main__":
    main()
