
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any, Dict, Optional


from pydantic import BaseModel, Field


class DISCOX(BaseModel):

    """
    DISCOX task class.
    """
    
    prompt_id:int
    prompt:str
    reference_list:str
    ori_text:str
    primary_domain:str= Field(alias="Primary_Domain")
    secondary_domain:str=Field(alias="Secondary_Domain")


    def extra_fields(self) -> Dict[str, Any]:
        return {
            "reference_list": self.reference_list,
            "prompt": self.prompt,
        }


def load_json(filename:str):
    if not filename.endswith(".json"):
        raise ValueError("filename must endswith .json")
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_tasks(filename:str):
    task=DISCOX
    data=load_json(filename)
    return [task(**item) for item in data]




#print(load_tasks("dataset/DISCOX-filtered.json"))