import os
from ollama import chat
from tools import (
    add_research_note,
    clear_research_notes,
    get_research_notes,
    get_verified_research_notes,
    read_webpage,
    search_web,
    search_academic_papers,
    verify_finding,
)


MODEL = "qwen3:4b"
TARGET_VERIFIED_NOTES = 2

AVAILABLE_TOOLS = {
    "search_web": search_web,
    "search_academic_papers": search_academic_papers,
    "read_webpage": read_webpage,
    "add_research_note": add_research_note,
    "get_research_notes": get_research_notes,
    "verify_finding": verify_finding,
}

TOOLS = [
    search_web,
    search_academic_papers,
    read_webpage,
    add_research_note,
    get_research_notes,
    verify_finding,
]


def generate_report(user_request: str, verified_notes: list) -> str:
    """Generates the final Markdown report from stored findings."""
    notes_summary = "\n\n".join(
        [f"Source: {n['source']}\nFinding: {n['finding']}\nEvidence: {n['evidence']}" for n in verified_notes]
    )

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a research analyst. Write a concise, professional research report in clean Markdown using headers and bullet points.",
            },
            {
                "role": "user",
                "content": f"Topic: {user_request}\n\nVerified Findings:\n{notes_summary}\n\nProduce a structured Markdown report.",
            },
        ],
    )
    return response.message.content


def research_agent(user_request: str, max_steps: int = 10, progress_callback=None):
    clear_research_notes()

    def emit(data):
        if progress_callback:
            progress_callback(data)

    emit({"type": "plan", "message": "Research plan created", "plan": "1. Search web\n2. Extract facts\n3. Synthesize report"})

    messages = [
        {
            "role": "system",
            "content": (
                "You are an automated research assistant. Follow this loop strictly:\n"
                "1. Search the web using multiple relevant queries.\n"
"2. Inspect search result titles, URLs, and snippets.\n"
"3. Read accessible webpages and extract relevant evidence.\n"
"4. If a website returns ACCESS_DENIED or a fetch error, "
"skip it and search for another accessible source.\n"
"5. Prefer peer-reviewed research, official reports, "
"and original studies.\n"
"6. Save findings with their source URL and supporting evidence.\n"
"7. Never invent sources, evidence, or statistics.\n"
"8. If searches fail, report a search failure rather than "
"claiming that no evidence exists.\n"
"9. Search the web using search_web.\n"
"10. Search academic papers using search_academic_papers.\n"
"11. Inspect titles, URLs, publication dates, and available abstracts.\n"
"12. Prefer relevant primary research and credible publications.\n"
"13. If a webpage returns ACCESS_DENIED, search for an "
"accessible version or another source.\n"
"14. Never interpret empty search results as proof that no evidence exists.\n"
"15. Never invent citations, findings, or statistics.\n"
"16. Record findings with source URLs and supporting evidence.\n"
"17. If the evidence is insufficient, state that clearly.\n"
"Do NOT read more than 2 pages without taking notes."
            ),
        },
        {"role": "user", "content": f"Research Topic: {user_request}"},
    ]

    for step in range(max_steps):
        verified = get_verified_research_notes()

        # Target met check: Stop execution early if target verified notes count is hit
        if len(verified) >= TARGET_VERIFIED_NOTES:
            emit({"type": "step", "step": step + 1, "message": "Target research findings met. Writing final report..."})
            report = generate_report(user_request, verified)
            emit({"type": "complete", "report": report})
            return report

        # Step cap fallback: Force finish if stuck beyond step 6 with at least 1 verified finding
        if step >= 6 and len(verified) >= 1:
            emit({"type": "step", "step": step + 1, "message": "Maximum research depth reached. Synthesizing available findings..."})
            report = generate_report(user_request, verified)
            emit({"type": "complete", "report": report})
            return report

        emit({"type": "step", "step": step + 1, "message": f"Agent step {step + 1}"})

        try:
            response = chat(model=MODEL, messages=messages, tools=TOOLS, options={"temperature": 0.1})
        except Exception as err:
            emit({"type": "error", "message": f"Execution error: {str(err)}"})
            return None

        messages.append(response.message)

        # Handle turns where the model produces plain text instead of tool calls
        if not response.message.tool_calls:
            notes = get_research_notes()
            if notes:
                latest = notes[-1]
                verify_finding(latest["source"], latest["finding"], latest["evidence"])
                emit({"type": "verification", "message": "Verified recorded note automatically"})
            else:
                messages.append({"role": "user", "content": "Please invoke a tool to continue research."})
            continue

        # Execute returned tool calls
        for tool_call in response.message.tool_calls:
            t_name = tool_call.function.name
            t_args = tool_call.function.arguments or {}

            emit({"type": "tool", "tool": t_name, "message": f"Using tool: {t_name}"})
            t_func = AVAILABLE_TOOLS.get(t_name)

            if t_func:
                try:
                    res = t_func(**t_args)
                    emit({"type": "tool_complete", "tool": t_name, "message": f"Completed: {t_name}"})
                except Exception as e:
                    res = f"Error executing tool: {str(e)}"
            else:
                res = "Tool not found."

            # Hard truncation on long webpage outputs to protect context length
            if t_name == "read_webpage" and isinstance(res, str):
                truncated_res = res[:1200] + "\n...[Webpage truncated to preserve context buffer]"
            else:
                truncated_res = str(res)

            messages.append({"role": "tool", "content": truncated_res, "tool_name": t_name})

            # Auto-verify notes right after saving them
            if t_name == "add_research_note":
                notes = get_research_notes()
                if notes:
                    latest = notes[-1]
                    verify_finding(latest["source"], latest["finding"], latest["evidence"])
                    emit({"type": "verification", "message": "Verified recorded note automatically"})

    # Final sweep
    verified = get_verified_research_notes()
    if verified:
        report = generate_report(user_request, verified)
        emit({"type": "complete", "report": report})
        return report

    emit({"type": "error", "message": "Loop ended without sufficient findings."})
    return None


def run_agent(request: str) -> str:
    """Helper entry point for synchronous execution."""
    return research_agent(request, max_steps=10)