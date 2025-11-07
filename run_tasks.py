# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm
import datetime
import os
from dotenv import load_dotenv
import glob
import traceback


from log import logger
from dataset.data import load_tasks,DISCOX
from eval.eval_res import metrics
from runs.run import litellm_generate

load_dotenv()


def run_task(task:DISCOX,model:str,judgemodel:str):
    """
    Run a single DISCOX task.
    Args:
        task (DISCOX): The DISCOX task to run.
        model (str): The model to be evaluated.
        judgemodel (str): The model to be used for judging.
        
    Returns:
        dict: A dictionary containing the task details, model output, domain scores, final score, and detailed judgement.
    """

    
    output=litellm_generate(messages=[{"role":"user","content":task.prompt}],model=model,judging_mode=False,max_tokens=20000)   
    judge_res,final_score,detailed_judgement=metrics(task,output,judgemodel)

    d = task.model_dump(by_alias=True)
    d["model_output"]=output
    d["domain_score"]=judge_res
    d["score"]=final_score
    d["detailed_judgement"]=detailed_judgement
    return d


def get_latest_jsonl_file(folder_path:str="./results",modelname:str="",taskname:str=""):
    # Check whether the model already evaluated
    all_files = glob.glob(os.path.join(folder_path, f"*{modelname}*{taskname}*.jsonl"))
    if not all_files:
        return None  #No corresponding file found

    # Get The latest file by creation time
    latest_file = max(all_files, key=os.path.getctime)
    return latest_file


def run_all_sync(tasks:list[DISCOX], model:str, judgemodel:str, concurrency:int=24):
    """
    Run all DISCOX tasks synchronously.
    Args:
        tasks (list[DISCOX]): The list of DISCOX tasks to run.
        model (str): The model to be evaluated.
        judgemodel (str): The model to be used for judging.
        concurrency (int, optional): The number of concurrent threads. Defaults to 24.
        
    Returns:
        res:dict: A dictionary containing the task details, model output, domain scores, final score, and detailed judgement.It will be
        saved to a json file.
        acc:float: The accuracy score of all tasks.
    """

    results = []


    modelname = model.replace("/", "-")
    tasksname = tasks[0].__class__.__name__
    last_run=get_latest_jsonl_file(modelname=modelname,taskname=tasksname)
    if last_run:
        with open(last_run,"r",encoding="utf-8") as f:
            lines=f.readlines()
            already_run_data = [json.loads(line)["prompt_id"] for line in lines if "prompt_id" in json.loads(line)]
            if len(already_run_data)<len(tasks):
                tasks=[task for task in tasks if task.prompt_id not in already_run_data]
                results=[json.loads(line) for line in lines]
            else:
                last_run=None
                logger.info(f"all tasks have been run in last experiments starting a new task,total:{len(tasks)}")
            
    else:
        logger.info(f"no previous run on this model,total:{len(tasks)}")



    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H-%M-%S")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    if not last_run:
        jsonl_file = os.path.join(output_dir, f"{formatted_time}-{modelname}-{tasksname}partial.jsonl")
        final_file = os.path.join(output_dir, f"{formatted_time}-{modelname}-{tasksname}result.json")
    else:
        jsonl_file=last_run
        final_file=last_run.replace("partial.jsonl","result.json")

    
    completed_counter = {"count": 0}
    counter_lock = threading.Lock()

    logger.info("start running")

    
    def wrapped_run(task):
        try:
            res = run_task(task, model, judgemodel)
            if "error" in res:
                logger.error(f"Task {task.prompt_id} failed with error: {res['error']}")
                raise RuntimeError("Unsupport task")
        except Exception as e:
            res = {"error": str(e), "traceback": traceback.format_exc()}
            logger.error(f"Task {task.prompt_id} exception:\n{traceback.format_exc()}")
        finally:
            with counter_lock:
                completed_counter["count"] += 1
        return res
    


    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(wrapped_run, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running tasks"):
            result = future.result()
            results.append(result)
            # Write result to jsonl file
            with open(jsonl_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Calcualte Accuracy
    valid = [v["score"] for v in results if "score" in v]
    acc = sum(valid) / (len(valid)*100) if valid else 0.0
    logger.info(f"Average_accuracy: {acc:.2%}")

    # Save final results
    
    res = {
        "average_accuracy": acc,
        "model": model,
        "detailed_results": results
    }
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {final_file}")

    return results, acc


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model with judge model on translation tasks")
    parser.add_argument("--filename", help="The filename of the task to be evaluated",default="dataset/DISCOX-filtered.json")
    parser.add_argument("--model", required=True, help="The model to be evaluated name or endpoint ID")
    parser.add_argument("--judgemodel", required=True, help="The judge model name or endpoint ID")
    parser.add_argument("--num_concurrency", type=int, default=32, help="Number of concurrent threads")
    return parser.parse_args()


if __name__=="__main__":  
    args = parse_arguments()  

    task = load_tasks(args.filename)
    model = args.model
    judgemodel = args.judgemodel
    run_all_sync(task, model, judgemodel,concurrency=args.num_concurrency)
