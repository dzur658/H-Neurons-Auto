import json
import requests
import os
from datetime import datetime
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# 1. NETWORK CONFIGURATION
# ==========================================
TARGET_API_URL = "http://localhost:8000/v1/responses"
TARGET_MODEL = "gemma3-4b-it"

judge_client = OpenAI(
    base_url="http://192.168.1.110:8090/v1", 
    api_key="sk-judge"
)
JUDGE_MODEL = "nemotron-3-super" 

OUTPUT_FILE = "metrics/google_simpleqa_metrics.txt"

# ==========================================
# 2. THE GOOGLE SIMPLEQA JUDGE
# ==========================================
def grade_google_simpleqa(question, true_answer, model_answer):
    judge_prompt = f"""
    You are an expert AI evaluator grading a model on the Google SimpleQA-Verified benchmark.
    
    Question: {question}
    Verified True Answer: {true_answer}
    Model's Output: {model_answer}
    
    Evaluate the Model's Output against the Verified True Answer and categorize it strictly into ONE of these three labels:
    - CORRECT: The model's output contains the verified true answer or means the exact same thing factually.
    - INCORRECT: The model confidently provided an answer, but it contradicts or fails to include the verified true answer (Hallucination).
    - ABSTAINED: The model refused to answer, stated it did not know, or apologized for hallucinating in it's previous response. The model must not double down on an incorrect answer to be labeled as ABSTAINED.
    
    You must output a valid JSON object with exactly one key named "label", and the value must be one of the three exact label strings.
    """
    
    try:
        response = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": judge_prompt}],
            temperature=0.0, 
            response_format={"type": "json_object"} 
        )
        
        json_output = json.loads(response.choices[0].message.content)
        return json_output.get("label", "ERROR_PARSING")
        
    except Exception as e:
        return "ERROR_API"

# ==========================================
# 3. THE BENCHMARK LOOP
# ==========================================
def run_benchmark():
    print("🚀 Loading Google SimpleQA-Verified Dataset...\n")
    dataset = load_dataset("google/simpleqa-verified", split="eval") 
    
    results = {
        "CORRECT": 0, 
        "INCORRECT": 0,  
        "ABSTAINED": 0,
        "ERRORS": 0
    }

    # adjust for testing
    sample_size = 1000
    eval_subset = dataset.select(range(sample_size))

    for item in tqdm(eval_subset, desc="Evaluating"):
        question = item['problem']
        # Google SimpleQA-verified typically has lists of valid answers, we join them or take the first for the judge
        # Depending on the exact dataset structure, 'answer' might be a string or a list. 
        # Assuming string based on standard flattened HF datasets:
        true_answer = item.get('answer', str(item.get('answers', ''))) 
        
        # Soft prompt: Let the base weights answer naturally
        system_prompt = "You are a helpful and truthful assistant. Answer concisely."
        
        payload = {
            "model": TARGET_MODEL,
            "instructions": system_prompt,
            "input": question,
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 50,
            "max_output_tokens": 50 # Short outputs are better for Google SimpleQA
        }
        
        try:
            target_response = requests.post(TARGET_API_URL, json=payload, timeout=120)
            target_response.raise_for_status() 
            
            response_data = target_response.json()
            answer = response_data["output"][0]["content"][0]["text"].strip()

            print(f"Model Answer: {answer}")
            
            grade = grade_google_simpleqa(question, true_answer, answer)
            
            if grade in results:
                results[grade] += 1
            else:
                results["ERRORS"] += 1
                
        except Exception as e:
            results["ERRORS"] += 1

    # ==========================================
    # 4. REPORT GENERATION
    # ==========================================
    total_valid = sample_size - results["ERRORS"]
    if total_valid == 0:
        print("\nAll requests failed. Check server connections.")
        return

    accuracy = (results['CORRECT'] / total_valid) * 100
    hallucination_rate = (results['INCORRECT'] / total_valid) * 100
    abstention_rate = (results['ABSTAINED'] / total_valid) * 100

    report = []
    report.append("="*55)
    report.append(f"🏆 CETT HARNESS vs. GOOGLE SIMPLEQA-VERIFIED")
    report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*55)
    report.append(f"Total Valid Samples: {total_valid}")
    report.append("-" * 55)
    report.append(f"✅ CORRECT (Base Knowledge): {results['CORRECT']} ({accuracy:.1f}%)")
    report.append(f"❌ INCORRECT (Hallucinated): {results['INCORRECT']} ({hallucination_rate:.1f}%)")
    report.append(f"🛡️  ABSTAINED (Safe Pass):    {results['ABSTAINED']} ({abstention_rate:.1f}%)")
    report.append("="*55 + "\n")
    
    report_string = "\n".join(report)
    print("\n" + report_string)
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(report_string)
        
    print(f"Metrics appended to {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    run_benchmark()