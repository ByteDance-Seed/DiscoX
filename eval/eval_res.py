# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import re


from runs.run import litellm_generate
from dataset.data import DISCOX
from eval.prompts import *
from log import logger


def extract_json_string(text: str) -> str:
    
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def safe_json_call(func):
    def wrapper(*args, retries=3, **kwargs):
        res = None
        for attempt in range(1, retries + 1):
            try:
                res=func(*args, **kwargs)
                return json.loads(res)
            except Exception as e:
                logger.info(f"Attempt {attempt}/{retries} failed when judging: {e}\n output result:{res}")

        return []
    return wrapper


def eval_result_by_llm(sp,up,judge_model,**kwargs):
    
    user_prompt=up.format(**kwargs)
    judge_input=[{"role":"system","content":sp},{"role":"user","content":user_prompt}]
    judgeres=litellm_generate(messages=judge_input,model=judge_model)
    
    return extract_json_string(judgeres)


@safe_json_call
def judge_accuracy(ori_text:str,response:str,judge_model):
    try:
        return eval_result_by_llm(accuracy_sp,up,judge_model,ori_text=ori_text,response=response)
    except Exception as e:
        
        return None


@safe_json_call
def judge_checkpoint(ori_text:str,response:str,checkpoints,judge_model:str):
    try:
        return eval_result_by_llm(checkpoints_sp,checkpoints_up,judge_model,ori_text=ori_text,response=response,checkpoints=checkpoints)
    except Exception as e:
        
        return None


@safe_json_call
def judge_fluency(ori_text:str,response:str,judge_model:str):
    try:
        return eval_result_by_llm(fluency_sp,up,judge_model,ori_text=ori_text,response=response)
    except Exception as e:
        
        return None


@safe_json_call
def judge_appropiate(ori_text:str,response:str,judge_model:str):
    try:
        return eval_result_by_llm(appropiate_sp,up,judge_model,ori_text=ori_text,response=response)
    except Exception as e:
        
        return None


def self_critique_judge(acc:list,flu:list,ckpt:list,appropiate:list,judge_model:str):
        res=eval_result_by_llm(self_critique_sp,self_critique_up,judge_model,accuracy_judge_result=acc,fluency_judge_result=flu,checkpoints_judge_result=ckpt,style_judge_result=appropiate)
        return res
    
    

@safe_json_call
def final_judge(sp:str,up:str,judge_model:str,**kwargs):
    try:
        return eval_result_by_llm(sp,up,judge_model,**kwargs)
    except Exception as e:
        return None


def metrics(task:DISCOX,response:str,judge_model:str):
    """
    Evaluate a task and a response by MetricS.
    Args:
        task (DISCOX): The task to be translated.
        response (str): The response string.
        judge_model (str): The judge model name.
        
    Returns:
        tuple: A tuple containing the scores of different dimensions (accuracy, checkpoints, fluency, appropiate), total score, and detailed judgement.
    """
    detailed_judgement={}
    instruct_res=eval_result_by_llm(instruct_sp,instruct_up,judge_model,prompt=task.ori_text,response=response)
    detailed_judgement["instruction_following"]=instruct_res
    match = re.search(r"是否存在问题：\s*([^\s])", instruct_res)
    if match:
       problem=match.group(1)
       if problem=="是":
           return (0,0,0),0,detailed_judgement


    acc_res=judge_accuracy(task.ori_text,response,judge_model)
    ckpt_res=judge_checkpoint(task.ori_text,response,task.reference_list,judge_model)
    flu_res=judge_fluency(task.ori_text,response,judge_model)
    appropiate_res=judge_appropiate(task.ori_text,response,judge_model)

    adjustment_res=self_critique_judge(acc_res,flu_res,ckpt_res,appropiate_res,judge_model)
    logger.info(f"adjustment_res:{adjustment_res}")
    acc_finalres=final_judge(accuracy_final_sp,accuracy_final_up,judge_model,accuracy_judge_result=acc_res,adjustment=adjustment_res)
    appropiate_finalres=final_judge(appropiate_final_sp,appropiate_final_up,judge_model,style_judge_result=appropiate_res,adjustment=adjustment_res)
    ckpt_finalres=final_judge(checkpoints_final_sp,checkpoints_final_up,judge_model,checkpoints_judge_result=ckpt_res,adjustment=adjustment_res)
    flu_finalres=final_judge(fluency_final_sp,fluency_final_up,judge_model,fluency_judge_result=flu_res,adjustment=adjustment_res)
    scores=calculate_score(acc_finalres,ckpt_finalres,flu_finalres,appropiate_finalres)
    detailed_judgement["initial_judgement"]={"accuracy":acc_res,"checkpoints":ckpt_res,"fluency":flu_res,"appropiate":appropiate_res}
    detailed_judgement["self_critique"]=adjustment_res
    detailed_judgement["accuracy"]=acc_finalres
    detailed_judgement["checkpoints"]=ckpt_finalres
    detailed_judgement["fluency"]=flu_finalres
    detailed_judgement["appropiate"]=appropiate_finalres
    
    return scores,sum(scores),detailed_judgement


def calculate_score(acc:list,ckpt:list,flu:list,appropiate:list):
    """
    Calculate the final score based on the final judge results 3 different dimensions.
    
    Args:
        acc (list): The accuracy final results.
        ckpt (list): The checkpoints final results.
        flu (list): The fluency final judge results.
        appropiate (list): The appropiate final judge results.
        
    Returns:
        tuple: A tuple of scores for three dimension (accuracy,fluency, appropiate).
    """


    ACCURACY_MAPPING = {
        "整体无问题": 0,
    '普通':     -5,  'Major':   -5,
    '严重': -10, 'Critical': -10,
    '非常严重': -50, 'Extremely Critical': -50,
}
    CORRECTNESS_MAPPING = {
    '错误': -5,
    '正确':  0,
}
    FLUENCY_MAPPING= {
    '无问题': 0,
    '有问题':  -2,
    "整体无问题": 0,    
}
    APPROPIATE_MAPPING = {
    '无问题':  0,
    '有问题': -5,
    "整体无问题": 0,
}
    # calculating accuracy score
    acc_score=60
    for item in acc:
        if "问题严重程度" in item and item["问题严重程度"] in ACCURACY_MAPPING:
            acc_score+=ACCURACY_MAPPING[item["问题严重程度"]]
        else:
            logger.warning(f"Unknown severity level: {item}. Defaulting to -5 points.")
            acc_score-=5
    for item in ckpt:
        #logger.info(item) 
        if "判断结果" in item and item["判断结果"] in CORRECTNESS_MAPPING:
            acc_score+=CORRECTNESS_MAPPING[item["判断结果"]]
        else:
            pass
            logger.warning(f"Unknown correctness level: {item}. Defaulting to consider it right.")
    acc_score=max(acc_score,0)
    # calculating fluency score
    flu_score=20
    for item in flu:
        if "问题严重程度" in item and item["问题严重程度"] in FLUENCY_MAPPING:
            flu_score+=FLUENCY_MAPPING[item["问题严重程度"]]
        else:
            logger.warning(f"Unknown severity level: {item}. Defaulting to -0 points.")
            flu_score-=0
    flu_score=max(flu_score,0)
    
    # calculating appropiate score
    appropiate_score=20
    for item in appropiate:
        if "问题严重程度" in item and item["问题严重程度"] in APPROPIATE_MAPPING:
            appropiate_score+=APPROPIATE_MAPPING[item["问题严重程度"]]
        else:
            logger.warning(f"Unknown severity level: {item}. Defaulting to -0 points.")
            appropiate_score+=0
    appropiate_score=max(appropiate_score,0)
    
    return acc_score,flu_score,appropiate_score

