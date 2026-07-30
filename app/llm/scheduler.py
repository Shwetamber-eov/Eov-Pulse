"""
scheduler.py

Priority-based scheduler for LangChain LLMs.

Responsibilities:
- Track RPM
- Track TPM
- Track RPD
- Select first available model by priority
- Wait if every model is busy
"""

import asyncio
import time

from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from app.llm.models import ModelConfig

CONTEXT_ERROR_MARKERS = (
    "context length",
    "maximum context",
    "context_length_exceeded",
    "too many tokens",
    "reduce the length",
    "token limit",
)

class ContextWindowExceeded(Exception):
    pass


# ============================================================
# Runtime State
# ============================================================

@dataclass
class ModelState:

    config: ModelConfig
    llm: object

    # timestamps of previous requests
    request_times: deque = field(default_factory=deque)

    # (timestamp, tokens)
    token_history: deque = field(default_factory=deque)

    # daily tracking
    daily_requests: int = 0
    daily_reset: float = field(default_factory=time.time)

    # health
    healthy: bool = True

    # thread safety
    lock: Lock = field(default_factory=Lock)

    # --------------------------------------------------------

    def _reset_daily(self):

        if time.time() - self.daily_reset >= 86400:
            self.daily_requests = 0
            self.daily_reset = time.time()

    # --------------------------------------------------------

    def _cleanup(self):

        now = time.time()

        # Sliding RPM window
        while self.request_times and now - self.request_times[0] >= 60:
            self.request_times.popleft()

        # Sliding TPM window
        while self.token_history and now - self.token_history[0][0] >= 60:
            self.token_history.popleft()

    # --------------------------------------------------------

    def mark_rate_limited(self):
        """
        Record an external 429 signal. Treated as a consumed
        request slot so we back off this model appropriately.
        """
        with self.lock:
            now = time.time()
            self.request_times.append(now)
            self.daily_requests += 1


    def available(self):

        with self.lock:

            self._reset_daily()
            self._cleanup()

            if not self.healthy:
                return False

            # RPM

            if len(self.request_times) >= self.config.rpm:
                return False

            # RPD

            if self.daily_requests >= self.config.rpd:
                return False

            # TPM

            if self.config.tpm is not None:

                used = sum(tokens for _, tokens in self.token_history)

                if used >= self.config.tpm:
                    return False

            return True

    # --------------------------------------------------------

    def record(self, tokens: int = 0):
        now = time.time()
        with self.lock:
            self.request_times.append(now)
            self.daily_requests += 1
            if self.config.tpm is not None:
                self.token_history.append((now, tokens))

    # --------------------------------------------------------


    def try_reserve(self):
        """
        Atomically check availability and, if available,
        immediately consume a slot. Returns True if reserved.
        """
        with self.lock:
            self._reset_daily()
            self._cleanup()

            if not self.healthy:
                return False
            if len(self.request_times) >= self.config.rpm:
                return False
            if self.daily_requests >= self.config.rpd:
                return False
            if self.config.tpm is not None:
                used = sum(tokens for _, tokens in self.token_history)
                if used >= self.config.tpm:
                    return False

            # Reserve immediately, still under the lock
            now = time.time()
            self.request_times.append(now)
            self.daily_requests += 1
            return True

    
    def wait_time(self):

        """
        Seconds until this model becomes usable.
        """
        with self.lock:
            waits = []

            now = time.time()

            if len(self.request_times) >= self.config.rpm:

                waits.append(
                    60 - (now - self.request_times[0])
                )

            if self.config.tpm is not None:

                used = sum(tokens for _, tokens in self.token_history)

                if (
                    used >= self.config.tpm
                    and len(self.token_history) > 0
                ):

                    waits.append(
                        60 - (now - self.token_history[0][0])
                    )

            if waits:
                return max(0, max(waits))

        return 0


# ============================================================
# Scheduler
# ============================================================

class ModelScheduler:

    def __init__(self, models):

        self.models = sorted(
            models,
            key=lambda m: m.config.priority
        )

    # --------------------------------------------------------

    def choose(self):

        """
        Returns the first available model.
        """
        while True:
            for model in self.models:
                if model.try_reserve():
                    print("chose :", model.config.name)
                    return model

            wait = min(model.wait_time() for model in self.models)
            print(f"All models busy. Waiting {wait:.1f}s")
            time.sleep(wait)

    # --------------------------------------------------------

    async def choose_async(self):

        while True:

            for model in self.models:

                if model.try_reserve():
                    print("async chose :", model.config.name)
                    return model

            wait = min(model.wait_time() for model in self.models)

            print(f"All models busy. Waiting {wait:.1f}s")

            await asyncio.sleep(wait)

    # ========================================================
    # Provider-specific token extraction
    # ========================================================

    @staticmethod
    def extract_tokens(model, response):

        metadata = getattr(response, "response_metadata", {})

        try:
            if model.config.provider == "google":
                # Prefer LangChain's standardized API
                usage = getattr(response, "usage_metadata", 0)

                if usage:
                    print("tokens used :",usage.get("total_tokens") or 
                        (usage.get("input_tokens", 0) + usage.get("output_tokens", 0)))
                    return usage.get("total_tokens") or (
                        usage.get("input_tokens", 0)
                        + usage.get("output_tokens", 0)
                    )

                # Fallback to provider-specific metadata
                usage = metadata.get("usage_metadata", {})
                print("tokens used :",usage.get("total_token_count")
                    or (
                        usage.get("prompt_token_count", 0)
                        + usage.get("candidates_token_count", 0)
                    ))
                return (
                    usage.get("total_token_count")
                    or (
                        usage.get("prompt_token_count", 0)
                        + usage.get("candidates_token_count", 0)
                    )
                )

            elif model.config.provider == "groq":   
#====================need to add different cases===============================
                usage = metadata.get("token_usage", {})
                return usage.get("total_tokens", 0)

        except Exception:
            return 0
    # ========================================================
    # Sync
    # ========================================================
    def invoke_with_model(self, model, messages):
        count=0
        while True:
            try:
                print("Using :", model.config.name)
                response = model.llm.invoke(messages)
                tokens = self.extract_tokens(model,response,)
                model.record(tokens)
                return response, model

            except Exception as e:
                text = str(e).lower()
                if any(marker in text for marker in CONTEXT_ERROR_MARKERS):
                # Not a capacity/rate problem — the chunk itself doesn't fit.
                    raise ContextWindowExceeded(str(e)) from e
                
                if count>=10:
                    raise RuntimeError(f"All models exhausted after {count} retries") from e
                count+=1
                print(text)
                if ("429" in text or "rate" in text):
                    print(f"{model.config.name} rate limited.")
                    model.mark_rate_limited()
                    # choose another available model
                    time.sleep(5)
                    model = self.choose()
                    continue
                raise

    def invoke(self, messages):
        model = self.choose()
        return self.invoke_with_model(model,messages,)

    # ========================================================
    # Async
    # ========================================================

    async def ainvoke_with_model(self,model,messages,):
        count=0
        while True:
            if count>=10:
                raise RuntimeError(f"All models exhausted after {count} retries")
            count+=1
            try:
                response = await model.llm.ainvoke(messages)
                tokens = self.extract_tokens(model,response,)
                model.record(tokens)
                return response

            except Exception as e:
                text = str(e).lower()
                if (
                    "429" in text
                    or "rate" in text
                ):

                    model.mark_rate_limited()
                    model = await self.choose_async()
                    continue
                raise

    async def ainvoke(self, messages):
        model = await self.choose_async()
        return await self.ainvoke_with_model(
            model,
            messages,
        )