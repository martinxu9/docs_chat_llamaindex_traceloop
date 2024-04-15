import os

import weaviate

from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from traceloop.sdk.decorators import workflow


query_engine: BaseQueryEngine

# Set these environment variables
WCS_URL = os.getenv("WCS_URL")
WCS_APIKEY = os.getenv("WCS_API_KEY", "")

INDEX_NAME = "ReflexLlamaindexTraceloopDemo"


def load_remote_vector_store():
    global query_engine
    client = weaviate.Client(
        url=WCS_URL, auth_client_secret=weaviate.AuthApiKey(api_key=WCS_APIKEY)
    )
    llm = OpenAI(model="gpt-4-0125-preview")
    service_context = ServiceContext.from_defaults(llm=llm)
    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name=INDEX_NAME,
    )
    loaded_index = VectorStoreIndex.from_vector_store(
        vector_store, service_context=service_context
    )
    query_engine = loaded_index.as_query_engine()


@workflow(name="reflex-chat-llamaindex-process-question")
async def process_question(chat):
    """Get the response from the API.

    Args:
        form_data: A dict with the current question.
    """
    global query_engine

    # Workaround: chat component currently uses list of dicts, llama-index Pydantic base model
    chat_history = [ChatMessage(**c) for c in chat.messages]

    # Start a new session.
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        chat_history=chat_history,
        verbose=True,
    )

    streaming_response = chat_engine.stream_chat(chat.last_user_message)
    # Stream the results, yielding after every word.
    for item in streaming_response.response_gen:
        chat.append_to_response(item)

        yield
