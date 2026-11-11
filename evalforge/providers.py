import json
import time
from abc import ABC, abstractmethod
from openai import OpenAI
from google import genai
from google.genai import types
from .models import ModelResponse
from .config import settings

class ProviderError(RuntimeError):
    pass

class BaseProvider(ABC):
    name = "base"
    def __init__(self, model: str | None = None):
        self.model = model or settings.model
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> ModelResponse:
        raise NotImplementedError

class OpenAICompatibleProvider(BaseProvider):
    name = "openai"
    def __init__(self, model=None, api_key=None, base_url=None, provider_name="openai"):
        super().__init__(model)
        key = api_key if api_key is not None else settings.openai_api_key
        url = base_url if base_url is not None else settings.openai_base_url
        if not key and not url:
            raise ProviderError("OPENAI_API_KEY is missing. Add it to .env, or use demo mode.")
        self.client = OpenAI(api_key=key or "ollama", base_url=url or None, timeout=settings.timeout)
        self.name = provider_name
    def generate(self, system_prompt, user_prompt):
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            elapsed = int((time.perf_counter() - start) * 1000)
            text = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            return ModelResponse(
                text=text, model=self.model, latency_ms=elapsed,
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
                metadata={"provider": self.name},
            )
        except Exception as exc:
            raise ProviderError(f"{self.name} generation failed: {exc}") from exc

class GeminiProvider(BaseProvider):
    name = "gemini"
    def __init__(self, model=None, api_key=None):
        super().__init__(model or settings.gemini_model)
        key = api_key if api_key is not None else settings.gemini_api_key
        if not key:
            raise ProviderError("GEMINI_API_KEY is missing. Add it to .env, or use demo mode.")
        self.client = genai.Client(api_key=key)
    def generate(self, system_prompt, user_prompt):
        start = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                ),
            )
            elapsed = int((time.perf_counter() - start) * 1000)
            usage = getattr(response, "usage_metadata", None)
            return ModelResponse(
                text=response.text or "", model=self.model, latency_ms=elapsed,
                input_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
                metadata={"provider": self.name},
            )
        except Exception as exc:
            raise ProviderError(f"Gemini generation failed: {exc}") from exc

class DemoProvider(BaseProvider):
    """Deterministic local provider: makes the app and test suite usable without API credentials."""
    name = "demo"
    def generate(self, system_prompt, user_prompt):
        start = time.perf_counter()
        lower = user_prompt.lower()
        if "refund" in lower:
            answer = "Customers can request a refund within 30 days of purchase. Refunds are processed to the original payment method within 5–7 business days."
        elif "password" in lower or "reset" in lower:
            answer = "Select 'Forgot password' on the sign-in page, enter your registered email, and follow the secure reset link. The link expires after 30 minutes."
        elif "shipping" in lower or "shipped" in lower:
            answer = "An order cannot be cancelled after it has shipped. You can refuse delivery or start a return after it arrives, subject to the return policy."
        elif "payment" in lower:
            answer = "If payment fails, check your payment details and available balance, then retry. If the issue continues, contact your bank or support. Do not share your full card number."
        elif "unknown" in lower or "not documented" in lower or "unsupported" in lower:
            answer = "I don't have enough information in the provided policy to answer that reliably. Please contact customer support for confirmation."
        else:
            answer = "I can help with that. Based on the supplied support policy, please check the relevant account or order details and contact support if you need account-specific assistance."
        return ModelResponse(text=answer, model=self.model, latency_ms=max(1, int((time.perf_counter()-start)*1000)), input_tokens=None, output_tokens=None, metadata={"provider":"demo"})

def build_provider(provider: str | None = None, model: str | None = None):
    provider = (provider or settings.provider).lower()
    if provider == "demo":
        return DemoProvider(model or "evalforge-demo-v1")
    if provider == "openai":
        return OpenAICompatibleProvider(model=model or settings.model)
    if provider == "ollama":
        return OpenAICompatibleProvider(
            model=model or settings.ollama_model,
            api_key="ollama",
            base_url=settings.ollama_base_url,
            provider_name="ollama",
        )
    if provider == "gemini":
        return GeminiProvider(model=model or settings.gemini_model)
    raise ProviderError(f"Unknown provider '{provider}'. Choose demo, openai, gemini, or ollama.")
