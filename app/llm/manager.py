"""
manager.py

High-level interface for interacting with LLMs.

Responsibilities:
- Create all LLM instances
- Create scheduler
- Expose invoke() and ainvoke()
"""

from models import MODEL_CONFIGS, create_llm
from scheduler import ModelScheduler, ModelState


class LLMManager:
    """Central manager for all configured language models."""

    def __init__(self):

        model_states = []

        for config in MODEL_CONFIGS:

            llm = create_llm(config)

            model_states.append(
                ModelState(
                    config=config,
                    llm=llm,
                )
            )

        self.scheduler = ModelScheduler(model_states)

    # --------------------------------------------------------

    def invoke(self, messages):
        """
        Send a request synchronously.

        Parameters
        ----------
        messages
            LangChain message list.

        Returns
        -------
        AIMessage
        """
        return self.scheduler.invoke(messages)

    # --------------------------------------------------------

    async def ainvoke(self, messages):
        """
        Async version.
        """
        return await self.scheduler.ainvoke(messages)


# ============================================================
# Singleton
# ============================================================

_manager = LLMManager()


# ============================================================
# Convenience Functions
# ============================================================

def invoke(messages):
    """
    Global invoke function.

    Example
    -------
    from llm.manager import invoke

    response = invoke(messages)
    """

    
    return _manager.invoke(messages)


async def ainvoke(messages):
    """
    Global async invoke function.
    """
    return await _manager.ainvoke(messages)


def get_manager():
    """
    Returns the singleton manager instance.
    """
    return _manager