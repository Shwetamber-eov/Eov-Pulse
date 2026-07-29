from chunker import TextChunker
from models import MODEL_CONFIGS, create_llm
from scheduler import ModelScheduler, ModelState


class LLMManager:

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

    # --------------------------------------------------
    # Normal prompts
    # --------------------------------------------------

    def invoke(self, messages):

        return self.scheduler.invoke(messages)

    async def ainvoke(self, messages):

        return await self.scheduler.ainvoke(messages)

    # --------------------------------------------------
    # Large document processing
    # --------------------------------------------------

    def invoke_document(
        self,
        pages,
        system_prompt: str,
    ):

        # Select the best available model
        model = self.scheduler.choose()

        # Use only 66% of context window
        safe_context = int(
            model.config.context_window * 0.66
        )

        chunker = TextChunker(safe_context)

        chunks = chunker.chunk(pages)

        previous_summary = ""

        for i, chunk in enumerate(chunks):

            if not previous_summary:

                human_prompt = chunk

            else:

                human_prompt = f"""
Previous Summary:
{previous_summary}

----------------------------------------

Current Document Chunk:
{chunk}

Update the previous summary using the
new information.

Return ONLY the updated JSON.
"""

            messages = [

                (
                    "system",
                    system_prompt,
                ),

                (
                    "human",
                    human_prompt,
                )

            ]

            response = self.scheduler.invoke_with_model(
                model,
                messages,
            )

            previous_summary = response.content

        return previous_summary


_manager = LLMManager()


def invoke(messages):

    return _manager.invoke(messages)


async def ainvoke(messages):

    return await _manager.ainvoke(messages)


def invoke_document(
    pages,
    system_prompt,
):

    return _manager.invoke_document(
        pages,
        system_prompt,
    )


def get_manager():

    return _manager