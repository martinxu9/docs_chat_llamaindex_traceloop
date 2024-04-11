from .rag_utils import init_vector_store, process_question

from rxconfig import config

from reflex_chat import chat

import reflex as rx


class State(rx.State):
    """The app state."""

    def load_engine(self):
        init_vector_store()


def index() -> rx.Component:
    return rx.center(
        chat(process=process_question),
        height="100vh",
    )


app = rx.App(theme=rx.theme(appearance="dark", accent_color="pink"))
app.add_page(
    index,
    title="Chat with docs | Reflex",
    on_load=State.load_engine,  # type:ignore
)
