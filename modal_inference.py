from __future__ import annotations

import hmac
import os
import re
from typing import Annotated

import modal
from fastapi import Header, HTTPException

app = modal.App("scholar-lens-summarizer")

MODEL_NAME = os.getenv("SCHOLAR_LENS_MODEL", "Qwen/Qwen2.5-3B-Instruct")
GPU_TYPE = os.getenv("SCHOLAR_LENS_GPU", "L4")
# Seconds to keep a warm container after the last request. Bump this (e.g.
# 600) right before recording a demo so the model never cold-starts on camera.
SCALEDOWN_WINDOW = int(os.getenv("SCHOLAR_LENS_SCALEDOWN", "90"))
# Keep the context small enough for the 3B model to start quickly and stay
# honest: Scholar Lens supplies the abstracts, then the model synthesizes them.
MAX_MODEL_LEN = 8192
MAX_PROMPT_TOKENS = 7000
SUMMARY_INPUT_CHAR_LIMIT = 60000
SUMMARY_INPUT_TOKEN_LIMIT = 15000
SUMMARY_CHUNK_TOKEN_LIMIT = 3500
SUMMARY_CHUNK_CHAR_LIMIT = 14000
SUMMARY_MAX_CHUNKS = 5
QUESTION_CHAR_LIMIT = 1200
QUESTION_TOKEN_LIMIT = 300
SYNTHESIS_CONTEXT_CHAR_LIMIT = 28000
SYNTHESIS_CONTEXT_TOKEN_LIMIT = 7000
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")

image = (
    # vLLM/flashinfer may JIT-compile CUDA kernels during warmup. The slim
    # Debian image lacks nvcc, so use a CUDA devel base with /usr/local/cuda.
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "vllm==0.8.5.post1",
        "fastapi[standard]",
    )
)


@app.cls(
    image=image,
    # Qwen2.5 3B keeps the small-model story load-bearing and deploys on a
    # modest GPU; override SCHOLAR_LENS_GPU for benchmark runs.
    gpu=GPU_TYPE,
    timeout=300,
    # Keep warm briefly for live demos without leaving an expensive GPU idle.
    scaledown_window=SCALEDOWN_WINDOW,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("scholar-lens-api"),
    ],
)
class Summarizer:
    @modal.enter()
    def load_model(self) -> None:
        """Load the model and tokenizer ONCE per container, not per request."""
        from vllm import LLM

        self.model = LLM(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.90,
        )

    def _generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        from vllm import SamplingParams

        prompt_tokens = self._rough_token_count(prompt)
        if prompt_tokens > MAX_PROMPT_TOKENS:
            raise ValueError(
                f"Prompt is too long for this endpoint ({prompt_tokens:,} rough tokens)."
            )

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

    @staticmethod
    def _rough_token_count(text: str) -> int:
        return len(TOKEN_PATTERN.findall(text or ""))

    def _input_limit_error(
        self,
        text: str,
        label: str,
        max_chars: int,
        max_tokens: int,
    ) -> str | None:
        if len(text) > max_chars:
            return f"{label} is too long. Keep it under {max_chars:,} characters."
        token_count = self._rough_token_count(text)
        if token_count > max_tokens:
            return (
                f"{label} is too long. Keep it under roughly {max_tokens:,} "
                f"tokens; this input is about {token_count:,} tokens."
            )
        return None

    def _chunk_summary_text(self, text: str) -> list[str]:
        chunks: list[str] = []
        current_words: list[str] = []
        current_chars = 0

        for word in text.split():
            proposed_chars = current_chars + len(word) + (1 if current_words else 0)
            proposed_tokens = len(current_words) + 1
            if (
                current_words
                and (
                    proposed_chars > SUMMARY_CHUNK_CHAR_LIMIT
                    or proposed_tokens > SUMMARY_CHUNK_TOKEN_LIMIT
                )
            ):
                chunks.append(" ".join(current_words))
                current_words = [word]
                current_chars = len(word)
            else:
                current_words.append(word)
                current_chars = proposed_chars

        if current_words:
            chunks.append(" ".join(current_words))
        return chunks[:SUMMARY_MAX_CHUNKS]

    def _summarize_text(self, text: str) -> str:
        chunks = self._chunk_summary_text(text)
        if len(chunks) <= 1:
            prompt = (
                "Summarize the following research paper context in 4-6 clear "
                "sentences. Cover the main contribution, methods, and key "
                "results/findings. If a Results / Findings section is present, "
                "use it as stronger evidence than the abstract.\n\n"
                f"Paper context:\n{text}"
            )
            return self._generate(prompt, max_new_tokens=250)

        chunk_summaries = []
        for index, chunk in enumerate(chunks, start=1):
            prompt = (
                "Summarize this section of a research paper in 2-3 concise "
                "sentences. Preserve concrete methods, results/findings, and limitations.\n\n"
                f"Section {index} of {len(chunks)}:\n{chunk}"
            )
            chunk_summaries.append(self._generate(prompt, max_new_tokens=180))

        combined = "\n\n".join(
            f"Section {index}: {summary}"
            for index, summary in enumerate(chunk_summaries, start=1)
        )
        final_prompt = (
            "Combine the section summaries below into one coherent 4-6 sentence "
            "summary of the paper. Avoid repetition and focus on the main "
            "contribution, methods, results/findings, and limitations.\n\n"
            f"{combined}"
        )
        return self._generate(final_prompt, max_new_tokens=280)

    def _require_auth(self, authorization: str | None) -> None:
        expected_token = os.getenv("SCHOLAR_LENS_MODAL_TOKEN", "").strip()
        if not expected_token:
            raise HTTPException(
                status_code=500,
                detail="Modal API token is not configured.",
            )

        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            raise HTTPException(status_code=401, detail="Unauthorized.")

        provided_token = authorization[len(prefix) :].strip()
        if not hmac.compare_digest(provided_token, expected_token):
            raise HTTPException(status_code=401, detail="Unauthorized.")

    @modal.fastapi_endpoint(method="POST", label="scholar-lens-summarizer-summarize-paper")
    def summarize_paper(
        self,
        data: dict,
        authorization: Annotated[str | None, Header()] = None,
    ) -> dict:
        self._require_auth(authorization)
        text = (data or {}).get("text", "")
        if not text:
            raise HTTPException(
                status_code=400,
                detail="No text provided in the request body.",
            )
        text = text.strip()
        limit_error = self._input_limit_error(
            text,
            "Text",
            SUMMARY_INPUT_CHAR_LIMIT,
            SUMMARY_INPUT_TOKEN_LIMIT,
        )
        if limit_error:
            raise HTTPException(status_code=400, detail=limit_error)

        try:
            summary = self._summarize_text(text)
        except Exception as exc:  # surface errors to the client instead of 500s
            print(f"summarize_paper generation failed: {exc}")
            return {"error": "Generation failed. Please try again shortly."}
        return {"summary": summary}

    @modal.fastapi_endpoint(method="POST", label="scholar-lens-summarizer-synthesize")
    def synthesize(
        self,
        data: dict,
        authorization: Annotated[str | None, Header()] = None,
    ) -> dict:
        """Answer a research question grounded ONLY in the supplied abstracts.

        Expects ``{"question": str, "context": str}`` where ``context`` is a
        block of numbered papers ([1], [2], ...). The model must cite those
        numbers, which keeps it from inventing sources.
        """
        self._require_auth(authorization)
        question = (data or {}).get("question", "")
        context = (data or {}).get("context", "")
        if not question or not context:
            raise HTTPException(
                status_code=400,
                detail="Both 'question' and 'context' are required.",
            )
        question = question.strip()
        context = context.strip()
        question_error = self._input_limit_error(
            question,
            "Question",
            QUESTION_CHAR_LIMIT,
            QUESTION_TOKEN_LIMIT,
        )
        if question_error:
            raise HTTPException(status_code=400, detail=question_error)
        context_error = self._input_limit_error(
            context,
            "Context",
            SYNTHESIS_CONTEXT_CHAR_LIMIT,
            SYNTHESIS_CONTEXT_TOKEN_LIMIT,
        )
        if context_error:
            raise HTTPException(status_code=400, detail=context_error)

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
            print(f"synthesize generation failed: {exc}")
            return {"error": "Generation failed. Please try again shortly."}
        return {"answer": answer}
