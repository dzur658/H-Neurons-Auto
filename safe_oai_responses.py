import os
# Force V1 to run the worker in the same memory space as FastAPI
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import gc
import joblib
import time
import torch
import uuid
import uvicorn
import argparse
from typing import List, Optional, Dict, Any, Union, Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Import your existing module
from scripts.auto.modernbert_qna import QnAModel

app = FastAPI(title="H-Neuron CETT Server")

engine: Optional[LLM] = None
cett_manager = None
model_tokenizer = None
modernbert: Optional[QnAModel] = None
classifier = None
configured_model_id: Optional[str] = None

# ==========================================
# 0. CETT Extraction Manager
# ==========================================
class VLLMCETTManager:
    def __init__(self, vllm_engine):
        # Drill down into vLLM's internal PyTorch architecture
        self.model = vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
        self.activations = []
        self.output_norms = []
        self.hooks = []
        self._register_hooks()
        self.weight_norms = self._get_weight_norms()

    def _register_hooks(self):
        def hook_fn(module, input, output):
            self.activations.append(input[0].detach())

            # extract the output tensor from the tuple vllm returns
            actual_output = output[0] if isinstance(output, tuple) else output

            self.output_norms.append(torch.norm(actual_output.detach(), dim=-1, keepdim=True))

        for name, module in self.model.named_modules():
            if 'down_proj' in name:
                self.hooks.append(module.register_forward_hook(hook_fn))

    def _get_weight_norms(self):
        norms = []
        for name, module in self.model.named_modules():
            if 'down_proj' in name:
                norms.append(torch.norm(module.weight.data, dim=0))
        device = next(self.model.parameters()).device
        return torch.stack(norms).to(device)

    def clear(self):
        self.activations.clear()
        self.output_norms.clear()

    def get_cett_tensor(self, use_abs=True, use_mag=True):
        """Returns normalized tensor of shape [layers, tokens, neurons]"""
        # get the total amount of layers in the model
        num_layers = len(self.hooks)

        # concatente prefill and generated tokens
        layer_acts = [torch.cat(self.activations[i::num_layers], dim=0) for i in range(num_layers)]
        layer_norms = [torch.cat(self.output_norms[i::num_layers], dim=0) for i in range(num_layers)]

        acts = torch.stack(layer_acts).to(self.weight_norms.device)
        norms = torch.stack(layer_norms).to(self.weight_norms.device)

        if use_abs: 
            acts = torch.abs(acts)
        if use_mag: 
            acts = acts * self.weight_norms.unsqueeze(1)
            
        return acts / (norms + 1e-8)


# arg parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Start the Safe LLM Response Server with integrated CETT hallucination detection.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model for config")
    parser.add_argument("--classifier_path", type=str, required=True, help="Path to the pre-trained H-Neuron classifier (pickle file)")
    parser.add_argument("--gpu_util", type=float, default=0.7, help="vLLM GPU memory utilization ratio")
    return parser.parse_args()

# ==========================================
# 1. API Schemas & Helpers
# ==========================================
class ResponsesRequest(BaseModel):
    model: str
    input: Union[str, List[Dict[str, Any]]]
    instructions: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    max_output_tokens: Optional[int] = 512
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"], Dict[str, Any]]] = "auto"
    metadata: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = False

class ResponseTextContent(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: List[Dict[str, Any]] = Field(default_factory=list)

class ResponseMessageItem(BaseModel):
    id: str
    type: Literal["message"] = "message"
    status: Literal["completed"] = "completed"
    role: Literal["assistant"] = "assistant"
    content: List[ResponseTextContent]

class ResponseFunctionCallItem(BaseModel):
    id: str
    type: Literal["function_call"] = "function_call"
    status: Literal["completed"] = "completed"
    call_id: str
    name: str
    arguments: str

def ensure_ready() -> None:
    if any(x is None for x in (engine, cett_manager, model_tokenizer, modernbert, classifier, configured_model_id)):
        raise HTTPException(status_code=503, detail="Service is not initialized")

def build_prompt(messages: List[Dict[str, str]]) -> str:
    if hasattr(model_tokenizer, "apply_chat_template"):
        return model_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prompt_lines: List[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        prompt_lines.append(f"{role}: {content}")
    prompt_lines.append("assistant:")
    return "\n".join(prompt_lines)

def normalize_user_text(input_payload: Union[str, List[Dict[str, Any]]]) -> str:
    if isinstance(input_payload, str):
        return input_payload

    chunks: List[str] = []
    for item in input_payload:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            chunks.append(content)
            continue
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in ("input_text", "text", "output_text") and isinstance(block.get("text"), str):
                    chunks.append(block["text"])
    return "\n".join([c for c in chunks if c]).strip()

def parse_tool_call(output_text: str, tools: Optional[List[Dict[str, Any]]], tool_choice: Any) -> Optional[ResponseFunctionCallItem]:
    if not tools:
        return None
    if tool_choice == "none":
        return None

    chosen_name: Optional[str] = None
    if isinstance(tool_choice, dict):
        chosen_name = tool_choice.get("name")

    if chosen_name is None and tool_choice == "required":
        for tool in tools:
            if tool.get("type") == "function" and isinstance(tool.get("name"), str):
                chosen_name = tool.get("name")
                break

    if chosen_name is None:
        return None

    args_payload = {"response": output_text.strip()}
    return ResponseFunctionCallItem(
        id=f"fc_{uuid.uuid4().hex}",
        call_id=f"call_{uuid.uuid4().hex}",
        name=chosen_name,
        arguments=str(args_payload).replace("'", '"')
    )

# ==========================================
# 2. The Core Synchronous Route
# ==========================================
@app.post("/v1/responses")
def create_response(request: ResponsesRequest) -> JSONResponse:
    ensure_ready()

    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported by this server")

    user_input = normalize_user_text(request.input)
    if not user_input:
        raise HTTPException(status_code=400, detail="input must include text content")

    messages = [
        {"role": "system", "content": request.instructions or ""},
        {"role": "user", "content": user_input}
    ]

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        max_tokens=request.max_output_tokens,
    )

    max_retries = 3
    attempts = 0
    
    while attempts < max_retries:
        formatted_prompt = build_prompt(messages)

        # 1. Clear previous hooks before generation
        cett_manager.clear()

        # 2. Generate text (Synchronous, blocks other requests to ensure pure tensor extraction)
        outputs = engine.generate([formatted_prompt], sampling_params)
        
        if not outputs or not outputs[0].outputs:
            raise HTTPException(status_code=500, detail="Generation produced no output")

        output_text = outputs[0].outputs[0].text
        prompt_token_ids = outputs[0].prompt_token_ids
        
        # 3. Span extraction
        span_result = modernbert.predict(question=user_input, context=output_text)

        print(f"Modernbert QnA Identified the following text as the answer in the LLM's response: {span_result['answer'] if span_result else 'None'}")

        is_hallucinating = False

        if span_result and span_result['score'] > 0.1:
            char_start, char_end = span_result["start"], span_result["end"]
            encoding = model_tokenizer(output_text, return_offsets_mapping=True, add_special_tokens=False)
            
            target_indices = [idx for idx, (start, end) in enumerate(encoding["offset_mapping"]) if end > char_start and start < char_end]
            
            if target_indices:
                # 4. Extract mathematically transformed CETT tensor [layers, tokens, neurons]
                cett_tensor = cett_manager.get_cett_tensor(use_abs=True, use_mag=True)
                
                # Apply prompt offset
                prompt_len_offset = len(prompt_token_ids)
                start_idx = target_indices[0] + prompt_len_offset
                end_idx = target_indices[-1] + 1 + prompt_len_offset

                # Clamp indices
                max_len = cett_tensor.shape[1]
                start_idx = max(0, min(start_idx, max_len - 1))
                end_idx = max(start_idx + 1, min(end_idx, max_len - 1))

                # Slice specific answer tokens across all layers
                answer_tensors = cett_tensor[:, start_idx:end_idx, :]

                # guard against NaN (frequently occurs if the answer is near the end of the generation)
                if answer_tensors.shape[1] == 0:
                    # slice to last generated token
                    answer_tensors = cett_tensor[:, -1:, :]
                
                # Aggregate exactly like the paper (mean across tokens)
                final_act = answer_tensors.mean(dim=1) 
                
                # Flatten into 1D array for Scikit-Learn
                flat_act = final_act.flatten().unsqueeze(0).cpu().float().numpy()
                
                predictions = classifier.predict(flat_act)
                is_hallucinating = 1 in predictions

        # Memory Cleanup
        cett_manager.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Routing Logic
        if not is_hallucinating:
            output_items: List[Dict[str, Any]] = []
            tool_call_item = parse_tool_call(output_text, request.tools, request.tool_choice)
            if tool_call_item is not None:
                output_items.append(tool_call_item.model_dump())
            else:
                message_item = ResponseMessageItem(
                    id=f"msg_{uuid.uuid4().hex}",
                    content=[ResponseTextContent(text=output_text.strip())],
                )
                output_items.append(message_item.model_dump())

            return JSONResponse({
                "id": f"resp_{uuid.uuid4().hex}",
                "object": "response",
                "created_at": int(time.time()),
                "status": "completed",
                "error": None,
                "incomplete_details": None,
                "instructions": request.instructions,
                "max_output_tokens": request.max_output_tokens,
                "model": configured_model_id,
                "output": output_items,
                "parallel_tool_calls": False,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "tool_choice": request.tool_choice,
                "tools": request.tools or [],
                "usage": {"input_tokens": None, "output_tokens": None, "total_tokens": None},
                "metadata": {
                    **(request.metadata or {}),
                    "requested_model": request.model,
                    "configured_model": configured_model_id,
                },
            })
        else:
            messages.append({"role": "assistant", "content": output_text})
            messages.append({
                "role": "user", 
                "content": f"[SYSTEM MESSAGE] Your previous response contained a hallucinated answer. The text specifically flagged as your answer, and the hallucination was: {span_result['answer'] if span_result else 'None'}. Helpful assistants are honest when they don't know an answer."
            })
            print(f"Hallucination detected in response! Retrying generation (attempt {attempts+1}/{max_retries})...")
            attempts += 1

    # Fallback response
    fallback_item = ResponseMessageItem(
        id=f"msg_{uuid.uuid4().hex}",
        content=[ResponseTextContent(text="I cannot create a verifiable answer to your prompt. Is this something an LLM is equipped to handle?")],
    )
    return JSONResponse({
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "error": None,
        "model": configured_model_id,
        "output": [fallback_item.model_dump()],
    })

# ==========================================
# 3. Server Execution
# ==========================================
if __name__ == "__main__":
    args = parse_args()
    configured_model_id = args.model_path

    print("Booting Synchronous vLLM Engine...")
    # CRITICAL: enforce_eager=True disables CUDA graphs so PyTorch hooks fire
    engine = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_util,
        enforce_eager=True 
    )

    print("Attaching CETT Manager...")
    cett_manager = VLLMCETTManager(engine)

    print("Loading Tokenizer and Watchdogs...")
    model_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    modernbert = QnAModel()
    
    classifier = joblib.load(args.classifier_path)

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)