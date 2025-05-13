import os
import json
import re
from mistralai import Mistral
import openai

# === Clients ===
# Mistral Client for Pro (Mistral API)
api_key = ""
mistral_client = Mistral(api_key=api_key)

# DeepSeek Client for Con agent
deepseek_client = openai.OpenAI(
    api_key="my_api_key",
    base_url="https://api.deepseek.com/v1"
)

# === Round 1: Pro ===
def generate_first_round_response_pro(case_text, options):
    system_prompt = "You are a helpful physician tasked with choosing the most likely diagnosis."
    user_prompt = f"""
Case:
{case_text}

Options:
{options}

Please select the most likely diagnosis from the options above and explain your reasoning in 3–4 medically accurate sentences. Return your answer in the following format:
Answer: <Option Letter>
Reasoning: <your step-by-step explanation>
"""
    response = mistral_client.chat.complete(
        model="mistral-large-latest",  # You can change the model as needed
        messages=[{"role": "user", "content": user_prompt.strip()}]
    )
    return response.choices[0].message.content.strip()

# === Round 1: Con ===
def generate_first_round_response_con(case_text, pro_text, options):
    system_prompt = "You are a critical physician challenging your colleague's diagnosis."
    user_prompt = f"""
Case:
{case_text}

Options:
{options}

Your colleague provided the following diagnosis and reasoning:
{pro_text}

Please propose a different option (choose a different letter from above) and explain your reasoning in 3–4 medically accurate sentences. Return your answer in the following format:
Answer: <Option Letter>
Reasoning: <your rebuttal and justification>
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.strip()}
    ]
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()

# === Round 2/3: Debate ===
def generate_agent_response(role, case_text, last_opponent_text=None, your_diagnosis=None):
    system_prompt = f"You are a {role} engaged in a clinical diagnostic debate."
    user_prompt = f"""
Case: {case_text}
Opponent's Previous Argument: {last_opponent_text}

Please continue the debate by defending your original diagnosis (**{your_diagnosis}**) with new reasoning.
Then, refute your opponent’s diagnosis with at least two comparative medical points.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()

# === Final: Consensus ===
def generate_consensus(case_text, pro_text, con_text):
    system_prompt = "You are a neutral senior medical consultant tasked with selecting the better diagnosis based on reasoning quality and clinical evidence."

    user_prompt = f"""
Case: {case_text}

Pro's Argument:
{pro_text}

Con's Argument:
{con_text}

Please:
1. First state your final diagnosis selection using the format: "Answer: A" (or B/C/D).
2. Summarize the key strengths and weaknesses of each physician's argument in 2–3 sentences each.
3. Justify your choice clearly in 2–3 sentences using medical reasoning 
"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.5,
        max_tokens=800
    )
    return response.choices[0].message.content.strip()

# === Inference Wrapper ===

def extract_answer(text):
    """
    Extracts the answer like 'Answer: B' from model output.
    """
    match = re.search(r"Answer:\s*([A-D])", text)
    return match.group(1).strip() if match else "?"

import json
import os

def run_all_benchmark_cases(json_path, pro_fn, con_fn, debate_fn, consensus_fn,
                            max_cases=None, start_index=0, save_path="results_partial.json"):
    with open(json_path, 'r') as f:
        benchmark = json.load(f)

    all_items = list(benchmark["medqa"].items())
    results = {}

    
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            results = json.load(f)

    count = 0

    for idx in range(start_index, len(all_items)):
        qid, entry = all_items[idx]
        if max_cases is not None and count >= max_cases:
            break

        if qid in results:
            print(f"⏭️ Skipping {qid} (already done)")
            continue

        question = entry["question"]
        options = "\n".join([f"{k}: {v}" for k, v in entry["options"].items()])
        ground_truth = entry["answer"]

        print(f"\n🚀 Running Case {qid}")
        try:
            # Round 1
            pro_text = pro_fn(question, options)
            con_text = con_fn(question, pro_text, options)

            pro_answer = extract_answer(pro_text)
            con_answer = extract_answer(con_text)

            # Round 2 & 3
            pro_text2 = debate_fn("pro physician", question, con_text, pro_answer)
            con_text2 = debate_fn("con physician", question, pro_text, con_answer)

            pro_text3 = debate_fn("pro physician", question, con_text2, pro_answer)
            con_text3 = debate_fn("con physician", question, pro_text2, con_answer)

            # Consensus
            consensus_text = consensus_fn(question, con_text3,pro_text3)
            consensus_answer = extract_answer(consensus_text)

            # Save result
            results[qid] = {
                "question": question,
                "pro_answer": pro_answer,
                "con_answer": con_answer,
                "consensus_answer": consensus_answer,
                "ground_truth": ground_truth,
                "pro_text_1": pro_text,
                "con_text_1": con_text,
                "pro_text_2": pro_text2,
                "con_text_2": con_text2,
                "pro_text_3": pro_text3,
                "con_text_3": con_text3,
                "consensus_text": consensus_text
            }

            # Write to disk immediately
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"✅ Saved result for {qid}")

            count += 1

        except Exception as e:
            print(f"❌ Error on case {qid}: {e}")
            continue

    print(f"\n🎉 Completed {count} new cases.")
    return results
