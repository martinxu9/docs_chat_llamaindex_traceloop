from .rag_utils import load_remote_vector_store, process_question

from rxconfig import config

from reflex_chat import chat

from traceloop.sdk import Traceloop

import reflex as rx


class State(rx.State):
    """The app state."""

    def load_engine(self):
        load_remote_vector_store()


def index() -> rx.Component:
    return rx.center(
        chat(process=process_question),
        height="100vh",
    )


Traceloop.init()
app = rx.App(theme=rx.theme(appearance="light", accent_color="purple"))
app.add_page(
    index,
    title="Chat with docs | Reflex",
    on_load=State.load_engine,  # type:ignore
)
