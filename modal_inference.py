import modal

app = modal.App("scholar-lens-summarizer")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.45.2",
        "accelerate==0.34.2",
        "bitsandbytes==0.44.1",
        "fastapi[standard]" # Add this since it uses FastAPI internally
    )
)

# FIXED HERE: Stack @app.function() and @modal.fastapi_endpoint()
@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    secrets=[modal.Secret.from_name("huggingface")]
)
@modal.fastapi_endpoint(method="POST") 
def summarize_paper(data: dict) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    text = data.get("text", "")
    if not text:
        return {"error": "No text provided in the request body."}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True
    )

    prompt = f"""Summarize the following research paper abstract in 4-5 clear sentences. Focus on the main contribution and results.

Abstract:
{text}

Summary:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Summary:" in summary:
        summary = summary.split("Summary:")[-1].strip()
    
    return {"summary": summary}