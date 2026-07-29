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

from models import ModelConfig


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

    def wait_time(self):

        """
        Seconds until this model becomes usable.
        """

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

                if model.available():
                    print("chose :", model.config.name)
                    return model

            wait = min(
                model.wait_time()
                for model in self.models
            )

            print(f"All models busy. Waiting {wait:.1f}s")

            time.sleep(wait)

    # --------------------------------------------------------

    async def choose_async(self):

        while True:

            for model in self.models:

                if model.available():
                    print("async chose :", model.config.name)
                    return model

            wait = min(
                model.wait_time()
                for model in self.models
            )

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
                usage = getattr(response, "usage_metadata", None)

                if usage:
                    print(usage.get("total_tokens") or 
                        (usage.get("input_tokens", 0) + usage.get("output_tokens", 0)))
                    return usage.get("total_tokens") or (
                        usage.get("input_tokens", 0)
                        + usage.get("output_tokens", 0)
                    )

                # Fallback to provider-specific metadata
                usage = metadata.get("usage_metadata", {})
                print(usage.get("total_token_count")
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

    def invoke(self, messages):

        while True:

            model = self.choose()

            try:
                print("chose in invoke function :", model.config.name)
                response = model.llm.invoke(messages)

                tokens = self.extract_tokens(
                    model,
                    response,
                )

                model.record(tokens)

                return response

            except Exception as e:

                text = str(e).lower()

                # 429

                if (
                    "429" in text
                    or "rate" in text
                ):

                    print(
                        f"{model.config.name} "
                        f"rate limited."
                    )

                    model.request_times.append(
                        time.time()
                    )

                    continue

                raise

    # ========================================================
    # Async
    # ========================================================

    async def ainvoke(self, messages):

        while True:

            model = await self.choose_async()

            try:
                print("chose in ainvoke function :", model.config.name)
                response = await model.llm.ainvoke(
                    messages
                )

                tokens = self.extract_tokens(
                    model,
                    response,
                )

                model.record(tokens)

                return response

            except Exception as e:

                text = str(e).lower()

                if (
                    "429" in text
                    or "rate" in text
                ):

                    print(
                        f"{model.config.name} "
                        f"rate limited."
                    )

                    model.request_times.append(
                        time.time()
                    )

                    continue

                raise