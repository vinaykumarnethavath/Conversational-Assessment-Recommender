from __future__ import annotations

import re
from typing import List

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import AssessmentRecommendation, ChatRequest, ChatResponse
from rag import AssessmentRAG

app = FastAPI(title="SHL Conversational Recommender", version="1.0.0")
rag = AssessmentRAG()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        messages = [msg.model_dump() for msg in payload.messages]
        if len(messages) > 20:
            return ChatResponse(
                reply="Please keep the conversation history to 20 messages or fewer.",
                recommendations=[],
                end_of_conversation=False,
            )

        latest_user_message = next(
            (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
            "",
        ).strip()

        if not latest_user_message:
            return ChatResponse(
                reply="Please provide your hiring requirements so I can recommend SHL assessments.",
                recommendations=[],
                end_of_conversation=False,
            )

        lowered = latest_user_message.lower()
        out_of_scope = re.search(
            r"\b(salary|compensation|visa|immigration|legal|lawsuit|terminate|fire|harass|medical|diagnos|therapy|resume|cv writing|cover letter)\b",
            lowered,
        )
        injection = re.search(
            r"\b(ignore (all|previous) instructions|system prompt|developer message|jailbreak|do anything now)\b",
            lowered,
        )
        if out_of_scope or injection:
            return ChatResponse(
                reply="I can only help with selecting and comparing SHL assessments from the SHL catalog. Please share the role, seniority, and what you want to assess (coding, cognitive ability, personality, leadership).",
                recommendations=[],
                end_of_conversation=False,
            )

        compare_pair = rag.detect_compare_request(latest_user_message)
        if compare_pair is not None:
            comparison = rag.compare(compare_pair[0], compare_pair[1])
            if comparison is None:
                return ChatResponse(
                    reply="I could not confidently match both assessments in the catalog. Please provide exact assessment names.",
                    recommendations=[],
                    end_of_conversation=False,
                )
            return ChatResponse(
                reply=comparison,
                recommendations=[],
                end_of_conversation=False,
            )

        state = rag.extract_state(messages)
        missing = rag.missing_clarifications(state)

        if missing:
            return ChatResponse(
                reply=rag.next_question(missing),
                recommendations=[],
                end_of_conversation=False,
            )

        items = rag.search(state, top_k=10)
        recommendations: List[AssessmentRecommendation] = []
        for item in items[:10]:
            url = (item.get("url") or "").strip()
            name = (item.get("name") or "").strip()
            test_type = (item.get("test_type") or "U").strip()
            if not name or not url:
                continue
            recommendations.append(
                AssessmentRecommendation(
                    name=name,
                    url=url,
                    test_type=test_type,
                )
            )
            if len(recommendations) >= 10:
                break

        if not recommendations:
            return ChatResponse(
                reply="I could not find strong matches in the current catalog. Please add more role details or required skills.",
                recommendations=[],
                end_of_conversation=False,
            )

        reply_text = rag.recommendation_reply(state, items)
        return ChatResponse(
            reply=reply_text,
            recommendations=recommendations,
            end_of_conversation=True,
        )
    except Exception:
        return ChatResponse(
            reply="Something went wrong while generating recommendations. Please try again with a shorter, clearer request.",
            recommendations=[],
            end_of_conversation=False,
        )
