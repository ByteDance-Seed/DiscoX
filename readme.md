# DISCOX

## Project Overview
DISCOX is a benchmark designed to evaluate the performance of LLMs on **discourse-level** and **expert-level translation tasks**.  
Unlike existing benchmarks that primarily focus on isolated sentences or general-domain texts, DISCOX emphasizes the ability of models to maintain discourse coherence, terminological precision, and domain-specific accuracy across long-form professional content.

This benchmark covers multiple domains (e.g., social sciences, natural sciences, humanities, applied disciplines, news, domain-specific scenarios, and literature & arts) with a wide range of translation challenges. It enables fine-grained evaluation of LLM outputs using criteria such as **accuracy, fluency, and appropriateness**.

## Getting Started

### 1. Install Dependencies
Make sure you are using **Python 3.9+**. Then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Set up your API key and endpoint in the `.env` file:
```bash
JUDGE_API_KEY=your_judgemodel_api_key_here
JUDGE_API_BASE=your_judgemodel_api_base_here
CANDIDATE_API_KEY=your_candidatemodel_api_key_here
CANDIDATE_API_BASE=your_candidatemodel_api_base_here
```

### 3. Run Evaluation Tasks
You can run tasks by specifying the target model and the judge model. For example:
```bash
python3 run_tasks.py --model openai/gpt4o-2024-11-20 --judgemodel azure/gemini-2.5-pro
```

## Example Use Case
- **Model Under Evaluation:** `openai/gpt4o-2024-11-20`  
- **Judge Model:** `azure/gemini-2.5-pro`  
This configuration runs translation tasks using GPT-4o and evaluates them with Gemini-2.5-Pro under the Metric-S evaluation framework.

---

## License
This project is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.