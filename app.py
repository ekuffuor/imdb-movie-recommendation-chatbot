import os
import gradio as gr
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from config.settings import OPENAI_API_KEY, FAISS_INDEX_PATH

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ---------------------------------------------------------------------------
# Load FAISS index
# ---------------------------------------------------------------------------
print("Loading FAISS index...")
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings_model,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
print("FAISS index loaded.")

# ---------------------------------------------------------------------------
# Retrieval chain (used as fallback / poster sourcing)
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_template("""
You are a helpful movie recommendation assistant. Based on the user's query,
provide movie recommendations from the retrieved results.

User Query: {query}

Retrieved Movies:
{context}

Provide a friendly, conversational response recommending these movies.
Include key details like genre, rating, director, and why they match the query.
""")


def format_docs(docs):
    parts = []
    for doc in docs:
        parts.append(
            f"Title: {doc.metadata['title']}\n"
            f"Genre: {doc.metadata['genre']}\n"
            f"Year: {doc.metadata['year']}\n"
            f"Director: {doc.metadata['director']}\n"
            f"Rating: {doc.metadata['rating']}\n"
            f"Cast: {doc.metadata['cast']}"
        )
    return "\n---\n".join(parts)


retrieval_chain = (
    {"context": retriever | format_docs, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

@tool
def search_movies_by_genre(genre: str) -> str:
    """Search for movies by genre (e.g. 'Action', 'Drama', 'Comedy')."""
    results = retriever.invoke(f"{genre} movies")
    lines = [
        f"{d.metadata['title']} ({d.metadata['year']}) — Rating: {d.metadata['rating']}"
        for d in results[:3]
    ]
    return "\n".join(lines)


@tool
def search_movies_by_director(director_name: str) -> str:
    """Search for movies directed by a specific director."""
    results = retriever.invoke(f"movies directed by {director_name}")
    lines = [
        f"{d.metadata['title']} ({d.metadata['year']}) — Rating: {d.metadata['rating']}"
        for d in results[:3]
        if director_name.lower() in d.metadata["director"].lower()
    ]
    return "\n".join(lines) if lines else f"No movies found for director {director_name}"


@tool
def search_high_rated_movies(min_rating: float = 7.5) -> str:
    """Search for highly rated movies above a minimum IMDb rating (default 7.5)."""
    results = retriever.invoke("highly rated acclaimed movies")
    lines = [
        f"{d.metadata['title']} ({d.metadata['year']}) — Rating: {d.metadata['rating']}"
        for d in results
        if d.metadata["rating"] >= min_rating
    ]
    return "\n".join(lines[:5]) if lines else "No movies found above that rating"


@tool
def search_movies_by_actor(actor_name: str) -> str:
    """Find movies featuring a specific actor."""
    results = retriever.invoke(f"movies featuring {actor_name}")
    lines = [
        f"{d.metadata['title']} ({d.metadata['year']}) — Rating: {d.metadata['rating']}"
        for d in results[:3]
        if actor_name.lower() in d.metadata["cast"].lower()
    ]
    return "\n".join(lines) if lines else f"No movies found with actor {actor_name}"


@tool
def search_movies_by_year_range(start_year: int, end_year: int) -> str:
    """Find movies released within a specific year range."""
    results = retriever.invoke(f"movies from {start_year} to {end_year}")
    lines = [
        f"{d.metadata['title']} ({d.metadata['year']})"
        for d in results
        if start_year <= d.metadata["year"] <= end_year
    ]
    return "\n".join(lines[:5]) if lines else f"No movies found between {start_year} and {end_year}"


# ---------------------------------------------------------------------------
# Agent / orchestrator
# ---------------------------------------------------------------------------
agent_tools = [
    search_movies_by_genre,
    search_movies_by_director,
    search_high_rated_movies,
    search_movies_by_actor,
    search_movies_by_year_range,
]

system_prompt = (
    "You are a movie recommendation orchestrator with access to specialized search tools.\n\n"
    "Available tools:\n"
    "- search_movies_by_genre: use when the user mentions a genre\n"
    "- search_movies_by_director: use when the user mentions a director\n"
    "- search_high_rated_movies: use when the user wants highly rated films\n"
    "- search_movies_by_actor: use when the user mentions an actor\n"
    "- search_movies_by_year_range: use when the user mentions years or decades\n\n"
    "Analyze the query, call the right tool(s), then give a friendly conversational response."
)

agent_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_react_agent(agent_model, agent_tools, messages_modifier=system_prompt)

# ---------------------------------------------------------------------------
# Gradio helpers
# ---------------------------------------------------------------------------

def build_poster_html(docs):
    html = "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:20px;padding:10px;'>"
    for doc in docs[:5]:
        title = doc.metadata["title"]
        poster_url = doc.metadata["poster"]
        rating = doc.metadata["rating"]
        year = doc.metadata["year"]
        html += (
            f"<div style='text-align:center;'>"
            f"<img src='{poster_url}' style='width:100%;max-width:150px;height:auto;"
            f"border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.2);' alt='{title}'>"
            f"<div style='margin-top:8px;font-size:12px;'>"
            f"<strong>{title}</strong><br>{year} | Rating: {rating}</div></div>"
        )
    html += "</div>"
    return html


def chatbot_response(user_query, chat_history):
    if not user_query.strip():
        return "", chat_history, ""

    try:
        result = agent.invoke({"messages": [HumanMessage(content=user_query)]})
        raw_content = result["messages"][-1].content
        bot_response = raw_content if isinstance(raw_content, str) else str(raw_content)
        retrieved_docs = retriever.invoke(user_query)
        poster_html = build_poster_html(retrieved_docs)
        chat_history.append((user_query, bot_response))
        return "", chat_history, poster_html
    except Exception as e:
        error_msg = f"Error: {e}"
        chat_history.append((user_query, error_msg))
        return "", chat_history, "<p>Error loading recommendations</p>"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="IMDb Movie Recommendation Chatbot") as demo:
    gr.Markdown("# IMDb Movie Recommendation Chatbot")
    gr.Markdown(
        "Powered by LangChain, OpenAI, and FAISS. "
        "Ask about genres, directors, actors, ratings, or years!"
    )

    chatbot_ui = gr.Chatbot(label="Movie Assistant", height=400)

    poster_display = gr.HTML(
        value="<p style='text-align:center;color:#999;padding:20px;'>Movie posters will appear here after you search.</p>",
        label="Recommended Movies",
    )

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="e.g. 'action movies with high ratings' or 'films by Christopher Nolan'",
            label="Your Question",
            scale=4,
        )
        submit_btn = gr.Button("Send", scale=1)

    clear_btn = gr.Button("Clear Chat")

    gr.Examples(
        examples=[
            "Recommend highly rated sci-fi movies",
            "What movies did Christopher Nolan direct?",
            "Show me romantic comedies from the 2000s",
            "Find action movies with a rating above 8",
            "I want a chill movie for a Friday night",
        ],
        inputs=user_input,
    )

    submit_btn.click(
        fn=chatbot_response,
        inputs=[user_input, chatbot_ui],
        outputs=[user_input, chatbot_ui, poster_display],
        api_name=False,
    )
    user_input.submit(
        fn=chatbot_response,
        inputs=[user_input, chatbot_ui],
        outputs=[user_input, chatbot_ui, poster_display],
        api_name=False,
    )
    clear_btn.click(
        fn=lambda: (
            [],
            "",
            "<p style='text-align:center;color:#999;padding:20px;'>Movie posters will appear here after you search.</p>",
        ),
        outputs=[chatbot_ui, user_input, poster_display],
        api_name=False,
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
