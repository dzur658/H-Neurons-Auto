import os
import json
import argparse
import time
from typing import List, Optional, Set

import torch
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer

# import modernbert qna model
from auto.modernbert_qna import QnAModel

def parse_args():
    parser = argparse.ArgumentParser(description="Extract answer tokens from consistent responses.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to samples files")
    parser.add_argument("--output_path", type=str, default="data/answer_tokens.jsonl", help="Path to save processed results")
    parser.add_argument("--tokenizer_path", type=str, default="data/activations", help="Path to the target model tokenizer")
    
    # add QnA BERT model judge option
    parser.add_argument("--judge_type", type=str, choices=["llm", "modernbert"], default="llm", help="How to judge correctness for token extraction")

    # LLM Extractor Config
    parser.add_argument("--api_key", type=str, help="OpenAI API Key")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="API Base URL")
    parser.add_argument("--llm_model", type=str, default="gpt-4o", help="LLM for extraction")
    
    return parser.parse_args()

# ==========================================
# LLM Prompt Templates
# ==========================================

USER_INPUT_TEMPLATE = """Question: {question}
Response: {response}
Tokenized Response: {response_tokens}
Please help extract the "answer tokens" from all tokens, removing all redundant information, and the tokens you return must be part of the input Tokenized Response list."""

EXAMPLE_MESSAGES = [
    {
        "role": "user", 
        "content": "Question: What is the correct name for the \"Flying Lady\" ornament on a Rolls Royce radiator.\nResponse: The correct name for the \"Flying Lady\" ornament on a Rolls Royce radiator is the Spirit of Ecstasy.\nTokenized Response: ['▁The', '▁correct', '▁name', '▁for', '▁the', '▁\"', 'F', 'lying', '▁Lady', '\"', '▁or', 'nament', '▁on', '▁a', '▁Roll', 's', '▁Roy', 'ce', '▁radi', 'ator', '▁is', '▁the', '▁Spirit', '▁of', '▁Ec', 'st', 'asy', '.']\nPlease help extract the \"answer tokens\" from all tokens, removing all redundant information, and the tokens you return must form a continuous segment of the input Tokenized Response list."
    },
    {
        "role": "assistant", 
        "content": "['▁the', '▁Spirit', '▁of', '▁Ec', 'st', 'asy']"
    },
]

# ==========================================
# Extraction Logic
# ==========================================

class AnswerTokenExtractor:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

        # Initialize OpenAI client if needed
        if self.args.judge_type == "llm":
            self.client = OpenAI(api_key=args.api_key, base_url=args.base_url)

        # initialze modernbert qna model if needed
        if self.args.judge_type == "modernbert":
            self.qna_model = QnAModel("rankyx/ModernBERT-QnA-base-squad")

    def get_tokenized_list(self, text: str) -> List[str]:
        # token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        # tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
        tokens = self.tokenizer.tokenize(text, add_special_tokens=False)

        return tokens

    def char_span_to_token_span(self, text: str, char_start: int, char_end: int) -> Optional[List[str]]:
        """Convert character span to token span using tokenizer offsets."""
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

        answer_tokens = []
        for idx, (start, end) in enumerate(encoding["offset_mapping"]):
            if end > char_start and start < char_end:
                # append decoded tokens
                # answer_tokens.append(self.tokenizer.decode([encoding["input_ids"][idx]]))

                tid = encoding["input_ids"][idx]
                raw_token = self.tokenizer.convert_ids_to_tokens(tid)
                answer_tokens.append(raw_token)

        # print(f"Answer Tokens: {answer_tokens}")
        return answer_tokens

    def extract_via_llm(self, question: str, response: str, tokens: List[str]) -> Optional[List[str]]:
        """Request LLM to select tokens from the tokenized list."""
        prompt = USER_INPUT_TEMPLATE.format(
            question=question,
            response=response,
            response_tokens=str(tokens)
        )
        
        for attempt in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.args.llm_model,
                    messages=EXAMPLE_MESSAGES + [{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                reply = completion.choices[0].message.content.strip().replace("'", "\"")
                extracted = json.loads(reply)
                
                # Validation: selected tokens must exist in original sequence
                if all(t in tokens for t in extracted):
                    return extracted
            except Exception as e:
                print(f"Extraction failed (attempt {attempt+1}): {e}")
                time.sleep(1)
        return None
    
    def modernbert_processing(self, question: str, response: str, tokens: List[str]) -> Optional[List[str]]:
        """Use ModernBERT-QnA to extract answer tokens."""
        qna_result = self.qna_model.predict(question, response)
        
        char_start = qna_result["start"]
        char_end = qna_result["end"]

        extracted_tokens = self.char_span_to_token_span(response, char_start, char_end)

        if all(t in tokens for t in extracted_tokens):
            return extracted_tokens
        return None

    def load_processed_ids(self) -> Set[str]:
        """Resume from existing output file."""
        if not os.path.exists(self.args.output_path): return set()
        ids = set()
        with open(self.args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                try: ids.update(json.loads(line).keys())
                except: continue
        return ids

    def run(self):
        processed_ids = self.load_processed_ids()

        # test
        # counter = 0
        
        with open(self.args.input_path, "r", encoding="utf-8") as f_in, \
            open(self.args.output_path, "a", encoding="utf-8") as f_out:
            
            for line in tqdm(f_in, desc="Processing tokens"):
                data = json.loads(line)
                qid = list(data.keys())[0]
                content = data[qid]

                if qid in processed_ids: continue

                # Ensure all 10 responses have the same judge outcome (true/false)
                judges = content["judges"]
                if len(set(judges)) != 1 or "uncertain" in judges or "error" in judges:
                    continue

                # Take the most frequent response as representative
                responses = content["responses"]
                rep_response = max(set(responses), key=responses.count)
                
                # Tokenization
                tokenized_list = self.get_tokenized_list(rep_response)

                if self.args.judge_type == "llm":
                    # LLM Extraction
                    answer_tokens = self.extract_via_llm(content["question"], rep_response, tokenized_list)
                else:
                    # ModernBERT-QnA Extraction
                    answer_tokens = self.modernbert_processing(content["question"], rep_response, tokenized_list)

                if answer_tokens:
                    result = {
                        qid: {
                            "question": content["question"],
                            "response": rep_response,
                            "tokenized_response": tokenized_list,
                            "answer_tokens": answer_tokens,
                            "judge": judges[0] # Consistently correct or consistently hallucinated
                        }
                    }
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()

                # testing
                # counter += 1
                # if counter >= 100:
                #     break

if __name__ == "__main__":
    args = parse_args()
    extractor = AnswerTokenExtractor(args)
    extractor.run()