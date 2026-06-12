import modal

app = modal.App("scholar-lens-summarizer")

MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
GPU_TYPE = "A100-80GB"
# Keep the demo context far below the model's 128k maximum so vLLM reserves
# less KV-cache memory and the endpoint starts more predictably.
MAX_MODEL_LEN = 32768

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.8.1",
        "mistral_common>=1.5.4",
        "fastapi[standard]",
    )
)


@app.cls(
    image=image,
    # Mistral Small 3.1 24B needs substantial VRAM in bf16/fp16. Use this
    # only for short judge demos, then move back down when the review is over.
    gpu=GPU_TYPE,
    timeout=300,
    # Keep warm briefly for live demos without leaving an expensive GPU idle.
    scaledown_window=90,
    secrets=[modal.Secret.from_name("huggingface")],
)
class Summarizer:
    @modal.enter()
    def load_model(self) -> None:
        """Load the model and tokenizer ONCE per container, not per request."""
        from vllm import LLM

        self.model = LLM(
            model=MODEL_NAME,
            tokenizer_mode="mistral",
            config_format="mistral",
            load_format="mistral",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.90,
        )

    def _generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        from vllm import SamplingParams

        messages = [{"role": "user", "content": prompt}]
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.15,
        )
        outputs = self.model.chat(
            messages,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text.strip()

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
