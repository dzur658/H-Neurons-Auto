import streamlit as st
import requests

st.set_page_config(page_title="Responses API Tester", layout="wide")
st.title("Open Responses API Diagnostic UI")

# --- Configuration Sidebar ---
with st.sidebar:
    st.header("Server Config")
    api_base = st.text_input("Base URL", value="http://localhost:8000/v1")
    model_id = st.text_input("Model ID", value="gemma3-4b-it")
    api_key = st.text_input("API Key (if required)", value="Bearer dummy-key", type="password")
    
    st.markdown("---")
    st.header("Behavior")
    # Added System Instructions text area
    system_message = st.text_area(
        "System Instructions", 
        value="You are a helpful and concise AI assistant.",
        help="Mapped to the top-level 'instructions' key in the payload."
    )
    
    st.markdown("---")
    st.header("Generation Parameters")
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
    top_k = st.slider("Top K", min_value=0, max_value=100, value=40, step=1)
    max_tokens_input = st.text_input("Max Tokens", value="512")
    
    st.markdown("---")
    st.markdown("**Endpoint Target:**")
    st.code(f"{api_base.rstrip('/')}/responses", language="text")

# --- Session State ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Chat Interface ---
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Send a test payload...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.history.append({"role": "user", "content": user_input})
    
    # --- Construct the Open Responses Payload ---
    payload = {
        "model": model_id,
        "instructions": system_message,  # Injected here at the root level
        "input": st.session_state.history,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k
    }
    
    if max_tokens_input.strip():
        try:
            payload["max_tokens"] = int(max_tokens_input.strip())
        except ValueError:
            st.sidebar.error("Max Tokens must be a valid integer. Sending without it.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key
    }
    
    endpoint = f"{api_base.rstrip('/')}/responses"

    # --- Fire the Request and Render Response ---
    with st.chat_message("assistant"):
        with st.spinner("Awaiting server response..."):
            try:
                response = requests.post(endpoint, json=payload, headers=headers)
                
                with st.expander("Raw Server Response (Debug)"):
                    st.json(response.json() if response.text else {"error": "Empty response body"})
                    st.text(f"Status Code: {response.status_code}")

                response.raise_for_status()
                data = response.json()
                
                # Parse the Responses API format
                if "output" in data and isinstance(data["output"], list):
                    messages = [item for item in data["output"] if item.get("type") == "message"]
                    
                    if messages:
                        assistant_msg = messages[-1]
                        
                        if isinstance(assistant_msg.get("content"), list):
                            reply_text = next((c["text"] for c in assistant_msg["content"] if c.get("type") == "output_text"), "No output_text found in content array.")
                        else:
                            reply_text = str(assistant_msg.get("content", "Unparseable content format."))
                            
                        st.markdown(reply_text)
                        st.session_state.history.append({"role": "assistant", "content": reply_text})
                    else:
                        st.error("Response 'output' array did not contain a 'message' block.")
                else:
                    st.error("Valid HTTP response, but missing the required 'output' array in the JSON schema.")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Connection Failed: {e}")