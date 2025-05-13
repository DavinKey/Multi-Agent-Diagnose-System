import openai
import time
import re
import os
import subprocess
import graphviz

# 🧠 GPT Client for all Judges
gpt_client = openai.OpenAI(api_key="")

# 🧠 DeepSeek Client for Pro and Con agents
deepseek_client = openai.OpenAI(
    api_key="",
    base_url="")

def generate_first_round_response_pro(case_text):
    user_prompt = f"""
Based on the following case, provide your most likely diagnosis and concise reasoning in 3-4 medically sound sentences.

{case_text}

Please state:
1. What diagnosis you suggest.
2. Your reasoning behind it (3-4 concise sentences).
"""
    messages = [
        {"role": "system", "content": "You are a supportive physician tasked with defending your proposed diagnosis."},
        {"role": "user", "content": user_prompt}
    ]

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()


def generate_first_round_response_con(case_text, pro_text):
    user_prompt = f"""
Your colleague believes the diagnosis is as follows:

=== Pro Physician Diagnosis ===
{pro_text}

However, you propose a different likely diagnosis. Please state:
1. What diagnosis you suggest instead.
2. Your reasoning behind it (3–4 concise sentences).
"""
    messages = [
        {"role": "system", "content": "You are a challenging physician tasked with proposing a medically sound alternative diagnosis."},
        {"role": "user", "content": user_prompt}
    ]
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()


def generate_agent_response(role, case_text, delta=1.0, last_opponent_text=None, your_diagnosis=None):
    if last_opponent_text and your_diagnosis:
        user_prompt = f"""
You are a {role}, participating in a structured clinical reasoning debate.

Please reply using the following two-section format:

B.x.1 {role}’s Defense:
Restate your suggested diagnosis (**{your_diagnosis}**) and provide **new or deeper reasoning** compared to the previous round. Do **not repeat earlier arguments**. Instead, add more medically detailed rationale (e.g., symptom timing, pathophysiology, prevalence, risk factors, diagnostic accuracy, clinical guidelines).

B.x.2 Refutation of the Opponent’s Diagnosis:
Acknowledge that the opponent's diagnosis is plausible, but explain **why your own is more likely**. Provide at least **two specific comparative points** (e.g., diagnostic specificity, typical symptom course, epidemiology, testing sensitivity).

---

Case:
{case_text}

Opponent's Previous Argument:
{last_opponent_text}
"""
        messages = [
            {"role": "system", "content": f"You are a {role} engaged in structured medical debate."},
            {"role": "user", "content": user_prompt}
        ]
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()

def check_agreement(role, your_diag, opponent_text):
                agree_prompt = f"""
You are a {role} reviewing your opponent's clinical argument below. Your original diagnosis was: "{your_diag}".

Opponent's Argument:
{opponent_text}

Do you now believe the opponent's diagnosis is more reasonable than yours?
Please answer with "Yes" or "No" and briefly justify your stance in 1–2 lines.
"""
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": agree_prompt}],
                    temperature=0.5,
                    max_tokens=200
                )
                return response.choices[0].message.content.strip()
def evaluate_with_critic(pro_text, con_text):
    eval_prompt = f"""
Please evaluate the following two physicians’ arguments using the CRIT framework: Claim, Reasoning, Informativeness, Trustworthiness. Each aspect should be scored from 0.0 to 1.0 with fine-grained differentiation:

Pro:
- Claim: X
- Reasoning: X
- Informativeness: X
- Trustworthiness: X

Con:
- Claim: X
- Reasoning: X
- Informativeness: X
- Trustworthiness: X

---

Here are the physician arguments:

=== Pro Physician ===
{pro_text}

=== Con Physician ===
{con_text}
"""
    judge_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-1106-preview"]
    all_judge_scores = []
    for model_name in judge_models:
        response = gpt_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.3,
            max_tokens=800
        )
        all_judge_scores.append((model_name, response.choices[0].message.content.strip()))
        time.sleep(1.0)
    return all_judge_scores

def generate_consensus(pro_text, con_text):
    user_prompt = f"""
You are a senior medical consultant evaluating the diagnostic arguments of two physicians.

Please do the following in your answer:
1. Summarize the key strengths and weaknesses of each physician's argument in 2–3 sentences each.
2. Decide which physician's diagnosis is more reasonable based on clinical logic, specificity, and evidence.
3. Justify your choice clearly in 2–3 sentences using medical reasoning.

End with this exact format:
Final Diagnosis: <your selected diagnosis>

=== Pro Physician ===
{pro_text}

=== Con Physician ===
{con_text}
"""
    messages = [
        {"role": "system", "content": "You are a senior medical consultant synthesizing arguments and delivering a clear judgment."},
        {"role": "user", "content": user_prompt}
    ]
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.5,
        max_tokens=700
    )
    return response.choices[0].message.content.strip()


def generate_simplified_mermaid_with_consensus_diagnosis(deepseek_client, pro_text, con_text, consensus_diagnosis):
    prompt = f"""
You are a medical visualization expert. Generate a clean and interpretable Mermaid flowchart using `graph TD` that illustrates the diagnostic reasoning paths of both physicians (Pro and Con) based on the debate.

Follow these strict instructions:
1. Use exactly 4 subgraphs in this order: Symptoms, Possible Diagnoses, Supporting Evidence, Final Decision.
2. Begin all diagnostic paths from the Symptoms subgraph.
3. Use two nodes under “Possible Diagnoses” — one for each physician's suggested diagnosis.
4. All reasoning and evidence should flow logically from Symptoms → Diagnoses → Evidence → Final Decision.
5. The Final Decision subgraph must contain exactly two nodes: 
   - One for the Pro‘s suggested diagnosis.
   - One for the Con’s suggested diagnosis.
6. Highlight the **correct consensus diagnosis node** using:  
   `style NODEID fill:#a3f7bf,stroke:#333,stroke-width:2px`  
   Ensure the NODEID matches the node ID used in the graph.
7. Keep the total number of nodes under 14 by merging redundant content.
8. Use full medical terms (no abbreviations, no emojis).
9. Output only raw Mermaid code. Do NOT include markdown, backticks, or explanations.

=== Pro Physician ===
{pro_text}

=== Con Physician ===
{con_text}

=== Consensus Diagnosis ===
{consensus_diagnosis}
"""
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()


def clean_mermaid_code(raw_code: str) -> str:
    # Remove any triple-backtick and optional mermaid marker
    code = re.sub(r"^```(?:mermaid)?", "", raw_code.strip(), flags=re.IGNORECASE).strip()
    code = re.sub(r"```$", "", code.strip())

    # Remove any stray HTML tags like <br> (sometimes returned by models)
    code = re.sub(r"<br\s*/?>", "\n", code)

    # Normalize line endings
    code = code.replace("\r\n", "\n").replace("\r", "\n")

    # Remove leading/trailing empty lines
    lines = [line.rstrip() for line in code.split("\n")]
    cleaned_lines = [line for line in lines if line.strip() != ""]

    return "\n".join(cleaned_lines).strip()

def render_graphviz_from_mermaid_text(mermaid_code: str, output_name="diagnosis_flowchart"):
    dot = graphviz.Digraph(format='png')
    dot.attr(rankdir='TB')

    nodes = {}
    edges = []

    for line in mermaid_code.splitlines():
        line = line.strip()
        if not line or line.startswith("graph") or line.startswith("subgraph") or line == "end":
            continue
        if "-->" in line:
            src, dst = [s.strip() for s in line.split("-->")]
            src_id = src.split("[")[0].strip()
            dst_id = dst.split("[")[0].strip()
            edges.append((src_id, dst_id))
            for part in [src, dst]:
                node_id = part.split("[")[0].strip()
                if "[" in part and "]" in part:
                    label = part.split("[", 1)[1].rsplit("]", 1)[0].strip()
                    nodes[node_id] = label
        elif "[" in line and "]" in line:
            node_id = line.split("[")[0].strip()
            label = line.split("[", 1)[1].rsplit("]", 1)[0].strip()
            nodes[node_id] = label

    for node_id, label in nodes.items():
        dot.node(node_id, label=label, shape="box")

    for src, dst in edges:
        dot.edge(src, dst)

    from IPython.display import Image, display

    filepath = dot.render(filename=output_name, cleanup=True)
    print(f"✅ Saved image: {filepath}")
    display(Image(filename=filepath))

def setup_case():
    return "Patient presents with itching, fatigue, lethargy, yellowish skin, dark urine, loss of appetite, abdominal pain, yellowing of the eyes, malaise, history of receiving a blood transfusion, and exposure to unsterile injections. Please determine the most likely diagnosis and explain your reasoning."

def run_debate(case_text=None):
    result = {
        "rounds": [],
        "consensus": "",
        "mermaid_code": "",
    }

    case = case_text
    delta = 1.0
    rounds = 3
    last_pro, last_con = "", ""
    first_pro, first_con = "", ""

    for r in range(rounds):
        round_info = {"round": r + 1}

        if r == 0:
            pro = generate_first_round_response_pro(case)
            con = generate_first_round_response_con(case, pro)
            first_pro, first_con = pro, con
        else:
            pro_agree = check_agreement("pro physician", last_pro, last_con)
            con_agree = check_agreement("con physician", last_con, last_pro)
            round_info["pro_agree"] = pro_agree
            round_info["con_agree"] = con_agree

            if "yes" in pro_agree.lower() and "yes" in con_agree.lower():
                pro = "I now agree with the Con physician's diagnosis based on the updated reasoning."
                con = "I now agree with the Pro physician's diagnosis based on the updated reasoning."
            elif "yes" in pro_agree.lower():
                pro = "I now agree with the Con physician's diagnosis based on the updated reasoning."
                con = generate_agent_response("con physician", case, delta, last_pro, last_con)
            elif "yes" in con_agree.lower():
                con = "I now agree with the Pro physician's diagnosis based on the updated reasoning."
                pro = generate_agent_response("pro physician", case, delta, last_con, last_pro)
            else:
                pro = generate_agent_response("pro physician", case, delta, last_con, first_pro)
                con = generate_agent_response("con physician", case, delta, last_pro, first_con)

        round_info["pro"] = pro
        round_info["con"] = con

        all_judges = evaluate_with_critic(pro, con)
        round_info["evaluation"] = [
            {"model": model, "score": score}
            for model, score in all_judges
        ]

        result["rounds"].append(round_info)
        last_pro, last_con = pro, con

    result["consensus"] = generate_consensus(last_pro, last_con)

    mermaid_code = generate_simplified_mermaid_with_consensus_diagnosis(
        deepseek_client, last_pro, last_con, result["consensus"]
    )
    cleaned_code = clean_mermaid_code(mermaid_code)
    result["mermaid_code"] = cleaned_code

    return result


import re

def run_inference_minimal_debate(case_text):
    result = {
        "case": case_text,
        "rounds": [],
        "consensus": "",
        "diagnosis": ""  
    }

    delta = 1.0
    rounds = 3
    last_pro, last_con = "", ""
    first_pro, first_con = "", ""

    for r in range(rounds):
        round_info = {"round": r + 1}

        if r == 0:
            pro = generate_first_round_response_pro(case_text)
            con = generate_first_round_response_con(case_text, pro)
            first_pro, first_con = pro, con
        else:
            pro = generate_agent_response("pro physician", case_text, delta, last_con, first_pro)
            con = generate_agent_response("con physician", case_text, delta, last_pro, first_con)

        round_info["pro"] = pro
        round_info["con"] = con
        result["rounds"].append(round_info)

        last_pro, last_con = pro, con

    consensus_text = generate_consensus(last_pro, last_con)
    result["consensus"] = consensus_text

   
    match = re.search(r"Final Diagnosis:\s*(.+)", consensus_text)
    if match:
        result["diagnosis"] = match.group(1).strip()

    return result