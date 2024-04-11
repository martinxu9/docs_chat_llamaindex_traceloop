from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.chat_engine import ContextChatEngine, CondenseQuestionChatEngine
from llama_index.core.base.llms.types import ChatMessage

query_engine: BaseQueryEngine


def init_vector_store(
    directory_path: str = "/Users/martinxu9/code/reflex/examples/reflex-web",
):
    global query_engine
    documents = SimpleDirectoryReader(
        input_dir=directory_path, recursive=True, required_exts=[".md"]
    ).load_data()

    print(f"Read number of docs from {directory_path}: {len(documents)}")

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()


async def process_question(chat):
    """Get the response from the API.

    Args:
        form_data: A dict with the current question.
    """
    global query_engine

    # Workaround: chat component currently uses list of dicts, llama-index Pydantic base model
    chat_history = [ChatMessage(**c) for c in chat.get_value(chat.messages)]

    # Start a new session.
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        chat_history=chat_history,
        verbose=True,
    )

    # Workaround: get the question
    for message in chat.messages[::-1]:
        if message["role"] == "user":
            question = message["content"]
            print(f"last question: {question}")

            streaming_response = chat_engine.stream_chat(question)

            # Stream the results, yielding after every word.
            for item in streaming_response.response_gen:
                chat.messages[-1]["content"] += item or ""
                yield
