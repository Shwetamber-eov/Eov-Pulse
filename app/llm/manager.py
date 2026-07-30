from app.llm.chunker import TextChunker
from app.llm.models import MODEL_CONFIGS, create_llm
from app.llm.scheduler import ModelScheduler, ModelState, ContextWindowExceeded

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

    def invoke_document(self, model, pages, system_prompt: str):
        print("inside invoke_document")
        remaining_pages = list(pages)
        previous_summary = ""

        while remaining_pages:

            safe_context = int(model.config.context_window * 0.80)
            chunker = TextChunker(safe_context)
            chunks = chunker.chunk(remaining_pages)
            print("len of chunks: ", len(chunks))
            for idx, chunk in enumerate(chunks):
                print("="*10)
                print("chunk :", idx+1)
                print("="*10)
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
                    ("system", system_prompt),
                    ("human", human_prompt),
                ]

                try:
                    response, new_model = self.scheduler.invoke_with_model(
                        model, messages
                    )
                except ContextWindowExceeded:
                    model = self.scheduler.choose()
                    remaining_pages = self._chunks_to_pages(chunks[idx:])
                    break

                previous_summary = response.content

                if new_model is not model:
                    # Model swapped (e.g. rate limit). Re-chunk what's
                    # left so later chunks respect the new context window.
                    model = new_model
                    remaining_pages = self._chunks_to_pages(chunks[idx + 1:])
                    break

            else:
                # Finished all chunks in this pass without a swap.
                remaining_pages = []

        return previous_summary


    @staticmethod
    def _chunks_to_pages(chunks):
        # Chunks are already page-sized text blocks; treat them
        # as the new "pages" list for re-chunking.
        return list(chunks)

    
    def invoke_auto(self,pages,system_prompt,):

        model = self.scheduler.choose()

        safe_context = int(
            model.config.context_window * 0.80
        )

        # Estimate total document size
        total_tokens = sum(
            TextChunker.estimate_tokens(page)
            for page in pages
        )

        if total_tokens <= safe_context:

            messages = [
                ("system", system_prompt),
                ("human", "\n\n".join(pages))
            ]

            response, model = self.scheduler.invoke_with_model(
                model,messages,
            )
            return response.content

        # Large document
        return self.invoke_document(
            model,
            pages,
            system_prompt,
        )


_manager = LLMManager()


def invoke(messages):

    return _manager.invoke(messages)


async def ainvoke(messages):

    return await _manager.ainvoke(messages)


def invoke_document(
    model,
    pages,
    system_prompt,
):

    return _manager.invoke_document(
        model,
        pages,
        system_prompt,
    )

def invoke_auto(pages,system_prompt,):

    return _manager.invoke_auto(pages,system_prompt)

def get_manager():

    return _manager