import wandb
import json
import os
from filelock import FileLock
from collections import defaultdict
from typing import Dict
import math
import numpy as np

class WandbLogger():
    def __init__(self, debug=False):
        self.debug = debug
        
    def init(self, project, entity, config, name):
        if self.debug:
            return
        wandb.init(project=project, entity=entity, config=config, name=name)
        
    def log(self, metrics):
        print(metrics)
        if self.debug:
            return
        wandb.log(metrics)
        
    def finish(self):
        if self.debug:
            return
        wandb.finish()
        
class DistributedWandbLogger():
    def __init__(self, original_num_episodes, num_jobs, job_index, tmp_dir="tmp", debug=False):
        self.original_num_episodes = original_num_episodes
        self.debug = debug
        self.num_jobs = num_jobs
        self.job_index = job_index
        if self.debug:
            return
        os.makedirs(tmp_dir, exist_ok=True)
        self.metrics_file = os.path.join(tmp_dir, f"metrics_{job_index}.json")
        self.metrics = []
        self.agg_metrics: Dict = defaultdict(float)
        self.tmp_dir = tmp_dir
        self.lock = FileLock(tmp_dir + ".lock")
        
    def close(self, project, entity, config, name):
        if self.debug:
            return
        print("Collected metrics:")
        print(self.metrics)
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f)
        with self.lock:
            filenames = os.listdir(self.tmp_dir)
            if len(filenames) == self.num_jobs:
                for filename in filenames:
                    if filename == f"metrics_{self.job_index}.json":
                        continue
                    with open(os.path.join(self.tmp_dir, filename), "r") as f:
                        self.metrics.extend(json.load(f))
                wandb.init(project=project, entity=entity, config=config, name=name)
                for elem in self.metrics:
                    # print(elem)
                    wandb.log(elem)
                    for m, v in elem.items():
                        if isinstance(v, dict):
                            for sub_m, sub_v in v.items():
                                self.agg_metrics[m + "/" + str(sub_m)] += sub_v
                        else:
                            self.agg_metrics[m] += v
                avg_metrics = {k: v / self.original_num_episodes for k, v in self.agg_metrics.items()}
                print("Average metrics:")
                print(avg_metrics)
                wandb.log(avg_metrics)
                wandb.finish()
        os.remove(self.lock)

    def log(self, metrics):
        print(metrics)
        if self.debug:
            return
        self.metrics.append(metrics)

class PINDistributedWandbLogger(DistributedWandbLogger):
    def close(self, project, entity, config, name):
        if self.debug:
            return
        print("Collected metrics:")
        print(self.metrics)
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f)
        with self.lock:
            filenames = os.listdir(self.tmp_dir)
            if len(filenames) == self.num_jobs:
                for filename in filenames:
                    if filename == f"metrics_{self.job_index}.json":
                        continue
                    with open(os.path.join(self.tmp_dir, filename), "r") as f:
                        self.metrics.extend(json.load(f))
                wandb.init(project=project, entity=entity, config=config, name=name)
                
                agg_metrics_per_category: Dict = defaultdict(dict)
                count_metrics: Dict = defaultdict(int)
                count_metrics_per_category: Dict = defaultdict(dict)
                
                for elem in self.metrics:
                    # print(elem)
                    wandb.log(elem)
                    if "category" in elem and elem["category"] not in agg_metrics_per_category:
                        agg_metrics_per_category[elem["category"]] = defaultdict(float)
                    for m, v in elem.items():
                        if isinstance(v, dict):
                            for sub_m, sub_v in v.items():
                                self.agg_metrics[m + "/" + str(sub_m)] += sub_v
                        else:
                            if m == "distance_to_goal":
                                if v == np.inf:
                                    continue
                            elif m == "spl":
                                if math.isnan(v):
                                    continue
                            elif m == "category":
                                continue
                            elif type(v) == list:
                                v = sum(v)
                            print(m, v)
                            self.agg_metrics[m] += v
                            if m == "correct_goal_detected":
                                if "found_goal" in elem and elem["found_goal"] == 1.0:
                                    count_metrics[m] += 1
                            else:
                                count_metrics[m] += 1
                            if "category" in elem:
                                if m not in agg_metrics_per_category[elem["category"]]:
                                    agg_metrics_per_category[elem["category"]][m] = v
                                    count_metrics_per_category[elem["category"]][m] = 1
                                else:
                                    agg_metrics_per_category[elem["category"]][m] += v
                                    count_metrics_per_category[elem["category"]][m] += 1

                avg_metrics = dict()
                for k, v in self.agg_metrics.items():
                    if k == "num_matched":
                        avg_metrics[k] = v
                    elif k in ("correct_matches", "category_matches", "generic_distractor_matches", "other_matches"):
                        avg_metrics[k] = v / self.agg_metrics["num_matched"]
                    else:
                        avg_metrics[k] = v / count_metrics[k]
                # avg_metrics = {k: v / self.original_num_episodes for k, v in self.self.agg_metrics.items()}
                avg_metrics["correct_goal_detected"] = self.agg_metrics[m] / (count_metrics["correct_goal_detected"] + 1e-6)
                for category in agg_metrics_per_category.keys():
                    for k, v in agg_metrics_per_category[category].items():
                        avg_metrics[f"{category}_{k}"] = v / (count_metrics_per_category[category][k] + 1e-6)

                print("Average metrics:")
                print(avg_metrics)
                wandb.log(avg_metrics)
                wandb.finish()
