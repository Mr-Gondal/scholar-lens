import modal

app = modal.App("scholar-lens-summarizer")

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.45.2",
        "accelerate==0.34.2",
        "bitsandbytes==0.44.1",
        "fastapi[standard]",
    )
)


@app.cls(
    image=image,
    gpu="A100",
    timeout=300,
    # Keep the container (and the loaded model) warm for 5 minutes after the
    # last request so repeat calls don't pay the cold-start cost again.
    scaledown_window=300,
    secrets=[modal.Secret.from_name("huggingface")],
)
class Summarizer:
    @modal.enter()
    def load_model(self) -> None:
        """Load the model and tokenizer ONCE per container, not per request."""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
        )

    def _generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens (avoids fragile string splitting).
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @modal.fastapi_endpoint(method="POST", label="scholar-lens-summarizer-summarize-paper")
    def summarize_paper(self, data: dict) -> dict:
        text = (data or {}).get("text", "")
        if not text:
            return {"error": "No text provided in the request body."}

        prompt = (
            "Summarize the following research paper abstract in 4-5 clear "
            "sentences. Focus on the main contribution and the key results.\n\n"
            f"Abstract:\n{text}"
        )
        try:
            summary = self._generate(prompt, max_new_tokens=250)
        except Exception as exc:  # surface errors to the client instead of 500s
            return {"error": f"Generation failed: {exc}"}
        return {"summary": summary}

    @modal.fastapi_endpoint(method="POST", label="scholar-lens-summarizer-synthesize")
    def synthesize(self, data: dict) -> dict:
        """Answer a research question grounded ONLY in the supplied abstracts.

        Expects ``{"question": str, "context": str}`` where ``context`` is a
        block of numbered papers ([1], [2], ...). The model must cite those
        numbers, which keeps it from inventing sources.
        """
        question = (data or {}).get("question", "")
        context = (data or {}).get("context", "")
        if not question or not context:
            return {"error": "Both 'question' and 'context' are required."}

        prompt = (
            "You are a meticulous research assistant. Using ONLY the numbered "
            "paper abstracts below, write a clear, synthesized answer to the "
            "question. Compare and contrast the findings where relevant. Cite "
            "every claim with the matching source number in square brackets, "
            "e.g. [1] or [2][3]. If the abstracts do not contain enough "
            "information to answer, say so plainly. Never invent sources or "
            "facts that are not in the abstracts.\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Synthesized answer (with [n] citations):"
        )
        try:
            answer = self._generate(prompt, max_new_tokens=450)
        except Exception as exc:
            return {"error": f"Generation failed: {exc}"}
        return {"answer": answer}
