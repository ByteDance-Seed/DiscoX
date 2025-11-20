# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional


import litellm


def litellm_generate(
    messages: list[dict],
    model: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    judging_mode: Optional[bool] = True,
    top_p: Optional[float] = None,
):
    """
    Generate model response using LiteLLM library.
    Args:
        messages (list[dict]): A list of message dictionaries.
        model (str): The name of the model to use for generation.
        temperature (Optional[float], optional): The temperature parameter for sampling.
            Defaults to None.
        max_tokens (Optional[int], optional): The maximum number of tokens to generate.
            Defaults to None.
        judging_mode (Optional[bool], optional): Whether to use judging mode (with
            temperature=0.0 and top_p=0.7). Defaults to True.
        top_p (Optional[float], optional): The top-p parameter.
            Defaults to None.
    Returns:
        Optional[str]: The generated response as a string, or None if an error occurs.
    """



    if judging_mode:
        temperature=0.0
        top_p=0.7
      
    try:
        if not judging_mode:
            api_base = os.environ["CANDIDATE_API_BASE"]
            api_key = os.environ["CANDIDATE_API_KEY"]
        else:
            api_base = os.environ["JUDGE_API_BASE"]
            api_key=os.environ["JUDGE_API_KEY"]

        
        kwargs = {
            "model": model,
            "top_p": top_p,
            "messages": messages,         
            "api_base": api_base,
            "api_key":api_key
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = litellm.completion(**kwargs)
        return response["choices"][0]["message"]["content"]

    except Exception as e:
        
        return None


if __name__=="__main__":        

      
    """data=[

      {
        "content": "翻译成英文：樱桃炸弹 UD2.0黑切 潮色主推\n专为 95/00 后的\nBarber 和美发师定制。",
        "role": "user"
      },
      {
        "content": "Cherry Bomb UD2.0 Blackout. The trendy color is highly recommended.\nSpecially customized for barbers and hairdressers born after 1995 and 2000.  ",
        "role": "assistant"
      },
      {
        "content": "说人话",
        "role": "user"
      },
      {
        "content": "Cherry Bomb UD2.0 Blackout version. It's our top-recommended trendy hair color.\nIt is specially designed for barbers and hairstylists who were born in the post-1995 and post-2000 generations. ",
        "role": "assistant"
      },
      {
        "content": "三色潮搭，颜值爆表。\n不止是推剪，更是创意伙伴。\n精密切割，动力强劲\n轻松塑造你的下一组爆款发型 。翻译",
        "role": "user"
      }
    ] 
    result=litellm_generate(data,model="openai/gpt-4o-1120")
        
    print(result)

        """
        
