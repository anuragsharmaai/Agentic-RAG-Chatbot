from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from app.config import settings
from app.tools.web_search import web_tool
from app.tools.pinecone_tool import vector_store
from app.safety import basic_content_filter, detect_prompt_injection, enforce_token_limit

class GraphState(TypedDict, total=False):
    question: str
    web_results: List[Dict[str, Any]]
    web_pages: List[Dict[str, Any]]
    rag_passages: List[Dict[str, Any]]
    draft: str
    summary: str
    sources: List[str]

def init_gemini():
    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model

# Tools

def tool_web_search(state: GraphState) -> GraphState:
    query = state.get("question", "")
    results = web_tool.search(query, num_results=state.get("max_web_results", 5))
    pages = []
    sources = []
    for r in results:
        link = r.get("link", "")
        text = web_tool.fetch_page_text(link)
        pages.append({"url": link, "text": text, "title": r.get("title", "")})
        sources.append(link)
    state["web_results"] = results
    state["web_pages"] = pages
    state["sources"] = sources
    return state

def tool_rag_search(state: GraphState) -> GraphState:
    query = state.get("question", "")
    k = state.get("max_rag_chunks", 5)
    matches = vector_store.similarity_search(query, k=k)
    passages = []
    for m in matches:
        passage = {
            "id": m.get("id"),
            "score": m.get("score"),
            "metadata": m.get("metadata", {}),
        }
        passages.append(passage)
    state["rag_passages"] = passages
    return state

# Agents

def research_agent(state: GraphState) -> GraphState:
    user_q = state.get("question", "")
    if detect_prompt_injection(user_q):
        state["draft"] = "Query flagged for possible prompt-injection. Please rephrase."
        return state
    state = tool_web_search(state)
    state = tool_rag_search(state)

    model = init_gemini()
    context_parts = []
    for p in state.get("web_pages", []):
        part = f"Source: {p.get('url')}\n{text_slice(p.get('text',''))}"
        context_parts.append(part)
    for r in state.get("rag_passages", []):
        meta = r.get("metadata", {})
        passage_text = meta.get("text", "")
        part = (
            f"RAG: id={r.get('id')} score={r.get('score')} "
            f"source={meta.get('source_id')} chunk={meta.get('chunk')}\n"
            f"{text_slice(passage_text)}"
        )
        context_parts.append(part)
    print(context_parts)

    prompt = f"You are a meticulous research assistant. Synthesize findings for: {user_q}\n\n"
    for cp in context_parts:
        prompt = prompt + cp + "\n\n"
    prompt = prompt + "Provide a structured note with key findings and citations."

    ok, filtered = basic_content_filter(prompt)
    if not ok:
        state["draft"] = filtered
        return state
    prompt = enforce_token_limit(filtered)

    try:
        response = model.generate_content(prompt)
        text = response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        text = f"Model error: {e}"

    state["draft"] = text
    return state

def text_slice(text: str, max_len: int = 2000) -> str:
    if text is None:
        return ""
    t = text
    if len(t) > max_len:
        return t[:max_len] + "..."
    return t

def summary_agent(state: GraphState) -> GraphState:
    model = init_gemini()
    draft = state.get("draft", "")
    question = state.get("question", "")
    sources = state.get("sources", [])

    sources_text = ""
    for s in sources:
        sources_text = sources_text + f"- {s}\n"

    prompt = (
        "Create a concise executive summary for busy executives.\n"
        "- Use bullet points.\n"
        "- Include a 2-sentence overview first.\n"
        "- Add a short risk/limitations section.\n"
        "- End with recommended next steps.\n\n"
        f"User question: {question}\n\n"
        f"Research draft: {draft}\n\n"
        f"Sources:\n{sources_text}"
    )

    ok, filtered = basic_content_filter(prompt)
    if not ok:
        state["summary"] = filtered
        return state
    prompt = enforce_token_limit(filtered)

    try:
        response = model.generate_content(prompt)
        text = response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        text = f"Model error: {e}"

    state["summary"] = text
    return state

# Build Graph

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("research_agent", research_agent)
    graph.add_node("summary_agent", summary_agent)

    graph.set_entry_point("research_agent")
    graph.add_edge("research_agent", "summary_agent")
    graph.add_edge("summary_agent", END)

    return graph.compile()

compiled_graph = build_graph()
