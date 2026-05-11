from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from embeddings import Embedder


@dataclass
class RequirementState:
    role: Optional[str]
    seniority: Optional[str]
    must_have_skills: List[str]
    wants_personality: bool
    wants_cognitive: bool
    wants_coding: bool
    wants_leadership: bool


class AssessmentRAG:
    def __init__(self, catalog_path: str = "data/assessments.json") -> None:
        self.catalog_path = Path(catalog_path)
        self.catalog: List[Dict[str, Any]] = self._load_catalog()
        self.embedder = Embedder()
        self.catalog_embeddings = self._build_embeddings(self.catalog)

    def _llm_enabled(self) -> bool:
        flag = os.getenv("LLM_ENABLED", "0").strip().lower() in {"1", "true", "yes"}
        return flag and bool(os.getenv("GROQ_API_KEY", "").strip())

    def _groq_client(self):
        try:
            from groq import Groq
        except Exception:
            return None
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            return None
        return Groq(api_key=api_key)

    def _groq_complete(self, prompt: str) -> Optional[str]:
        if not self._llm_enabled():
            return None
        client = self._groq_client()
        if client is None:
            return None
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip() or "llama-3.1-8b-instant"
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an SHL assessment assistant. Be concise, accurate, and never invent assessment names or URLs."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=250,
            )
            content = resp.choices[0].message.content if resp and resp.choices else None
            if content:
                return content.strip()
        except Exception:
            return None
        return None

    def _load_catalog(self) -> List[Dict[str, Any]]:
        candidates = [
            Path("data/shl_catalog.json"),
            self.catalog_path,
            Path("data/assessments_scraped.json"),
        ]
        data: List[Dict[str, Any]] = []
        for path in candidates:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8-sig") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list):
                data = loaded
                break
        return [self._normalize_item(item) for item in data if isinstance(item, dict)]

    def _normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        keys = self._ensure_list(item.get("keys"))
        category = str(item.get("category") or ", ".join(keys) or "General")
        skills = self._ensure_list(item.get("skills"))
        if not skills:
            skills = keys.copy()

        normalized: Dict[str, Any] = {
            "name": item.get("name", "Unknown Assessment"),
            "url": item.get("url") or item.get("link") or "",
            "category": category,
            "skills": skills,
            "description": item.get("description") or "",
            "duration_minutes": self._parse_duration_minutes(
                item.get("duration_minutes") or item.get("duration") or item.get("duration_raw")
            ),
            "job_levels": self._ensure_list(item.get("job_levels")),
            "languages": self._ensure_list(item.get("languages")),
            "keys": keys,
            "test_type": item.get("test_type") or self._infer_test_type(keys, category),
        }
        return normalized

    def _ensure_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return []
            return [part.strip() for part in cleaned.split(",") if part.strip()]
        return []

    def _parse_duration_minutes(self, duration: Any) -> Optional[int]:
        if duration is None:
            return None
        if isinstance(duration, int):
            return duration
        text = str(duration).lower()
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1))
        return None

    def _infer_test_type(self, keys: List[str], category: str) -> str:
        source = f"{' '.join(keys)} {category}".lower()
        if "knowledge" in source or "skill" in source:
            return "K"
        if "personality" in source or "behavior" in source:
            return "P"
        if "ability" in source or "aptitude" in source:
            return "A"
        if "simulation" in source:
            return "S"
        if "competenc" in source:
            return "C"
        return "U"

    def _doc_text(self, item: Dict[str, Any]) -> str:
        return (
            f"{item.get('name', '')}. "
            f"Category: {item.get('category', '')}. "
            f"Skills: {', '.join(item.get('skills', []))}. "
            f"Levels: {', '.join(item.get('job_levels', []))}. "
            f"Keys: {', '.join(item.get('keys', []))}. "
            f"{item.get('description', '')}"
        )

    def _build_embeddings(self, catalog: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        if not catalog:
            return None
        texts = [self._doc_text(item) for item in catalog]
        return self.embedder.encode(texts)

    def extract_state(self, messages: List[Dict[str, str]]) -> RequirementState:
        user_text = " ".join(
            msg["content"] for msg in messages if msg.get("role") == "user"
        ).lower()

        role = self._extract_role(user_text)
        seniority = self._extract_seniority(user_text)
        skills = self._extract_skills(user_text)

        wants_personality = any(k in user_text for k in ["personality", "opq", "culture fit", "behavioral"]) and not any(k in user_text for k in ["drop personality", "no personality", "without personality", "remove personality"])
        wants_cognitive = any(k in user_text for k in ["cognitive", "aptitude", "reasoning", "general ability"]) and not any(k in user_text for k in ["drop cognitive", "no cognitive", "without cognitive", "remove cognitive", "drop aptitude", "no aptitude"])
        wants_coding = any(k in user_text for k in ["coding", "programming", "developer", "engineer", "technical"]) and not any(k in user_text for k in ["drop coding", "no coding", "without coding", "remove coding", "drop technical", "no technical"])
        wants_leadership = any(k in user_text for k in ["leadership", "manager", "team lead", "people management"]) and not any(k in user_text for k in ["drop leadership", "no leadership", "without leadership", "remove leadership"])

        return RequirementState(
            role=role,
            seniority=seniority,
            must_have_skills=skills,
            wants_personality=wants_personality,
            wants_cognitive=wants_cognitive,
            wants_coding=wants_coding,
            wants_leadership=wants_leadership,
        )

    def missing_clarifications(self, state: RequirementState) -> List[str]:
        missing = []
        if not state.role:
            missing.append("role")
        if not state.seniority:
            missing.append("seniority")
        if not (state.wants_coding or state.wants_personality or state.wants_cognitive or state.wants_leadership):
            missing.append("assessment_focus")
        return missing

    def next_question(self, missing: List[str]) -> str:
        if self._llm_enabled():
            prompt = (
                "Ask exactly one short clarification question to collect missing hiring requirements. "
                f"Missing fields: {', '.join(missing)}. "
                "Do not mention internal field names."
            )
            completion = self._groq_complete(prompt)
            if completion:
                return completion
        if "role" in missing:
            return "What role are you hiring for (for example Java developer, analyst, or sales manager)?"
        if "seniority" in missing:
            return "What experience level are you hiring for (entry, mid, or senior)?"
        return "Do you want technical coding tests, cognitive aptitude tests, personality tests, leadership tests, or a combination?"

    def recommendation_reply(self, state: RequirementState, items: List[Dict[str, Any]]) -> str:
        if not items:
            return "Here are the most relevant SHL assessments based on your requirements."
        if not self._llm_enabled():
            return "Here are the most relevant SHL assessments based on your requirements."

        item_lines = []
        for item in items[:10]:
            name = (item.get("name") or "").strip()
            url = (item.get("url") or "").strip()
            test_type = (item.get("test_type") or "").strip()
            if name and url:
                item_lines.append(f"- {name} ({test_type}) {url}")

        prompt = (
            "Write a concise recruiter-facing response explaining why the following SHL assessments fit the user's request. "
            "You must only reference assessments from the provided list. Do not invent anything. "
            "End with a short question offering refinement (e.g., add/remove personality/cognitive/coding).\n\n"
            f"Role: {state.role or 'unknown'}\n"
            f"Seniority: {state.seniority or 'unknown'}\n"
            f"Must-have skills: {', '.join(state.must_have_skills) or 'none'}\n"
            f"Requested focus: "
            f"coding={state.wants_coding}, cognitive={state.wants_cognitive}, personality={state.wants_personality}, leadership={state.wants_leadership}\n\n"
            "Assessments (name, type, url):\n" + "\n".join(item_lines)
        )
        completion = self._groq_complete(prompt)
        return completion or "Here are the most relevant SHL assessments based on your requirements."

    def compare_reply(self, left: Dict[str, Any], right: Dict[str, Any]) -> str:
        if not self._llm_enabled():
            return (
                f"{left['name']} is a {left['category']} assessment ({left.get('duration_minutes') or 'NA'} mins) "
                f"focused on {', '.join(left['skills'][:3])}. "
                f"{right['name']} is a {right['category']} assessment ({right.get('duration_minutes') or 'NA'} mins) "
                f"focused on {', '.join(right['skills'][:3])}. "
                "Choose the first when role fit relies on its skill profile, and the second when the alternate profile is more relevant."
            )

        def fmt(item: Dict[str, Any]) -> str:
            return (
                f"Name: {item.get('name','')}\n"
                f"Category: {item.get('category','')}\n"
                f"Test type: {item.get('test_type','')}\n"
                f"Duration minutes: {item.get('duration_minutes') or 'NA'}\n"
                f"Skills: {', '.join(item.get('skills', [])[:10])}\n"
                f"URL: {item.get('url','')}\n"
            )

        prompt = (
            "Compare these two SHL assessments using only the provided catalog fields. "
            "Explain differences in category/focus and when to choose each. Be concise.\n\n"
            "Assessment A:\n" + fmt(left) + "\n"
            "Assessment B:\n" + fmt(right)
        )
        completion = self._groq_complete(prompt)
        return completion or (
            f"{left['name']} is a {left['category']} assessment ({left.get('duration_minutes') or 'NA'} mins) "
            f"focused on {', '.join(left['skills'][:3])}. "
            f"{right['name']} is a {right['category']} assessment ({right.get('duration_minutes') or 'NA'} mins) "
            f"focused on {', '.join(right['skills'][:3])}. "
            "Choose the first when role fit relies on its skill profile, and the second when the alternate profile is more relevant."
        )

    def detect_compare_request(self, text: str) -> Optional[Tuple[str, str]]:
        lowered = text.lower().strip()
        patterns = [
            r"difference between (.+?) and (.+?)[\?]?$",
            r"compare (.+?) and (.+?)[\?]?$",
            r"(.+?) vs\.? (.+?)[\?]?$",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                return match.group(1).strip(), match.group(2).strip()
        return None

    def compare(self, left_name: str, right_name: str) -> Optional[str]:
        left = self._fuzzy_find(left_name)
        right = self._fuzzy_find(right_name)
        if not left or not right:
            return None
        return self.compare_reply(left, right)

    def _fuzzy_find(self, text: str) -> Optional[Dict[str, Any]]:
        query = text.lower()
        aliases = {
            "opq": "occupational personality questionnaire",
            "gsa": "global skills assessment",
            "gat": "general ability",
        }
        query = aliases.get(query, query)
        best = None
        best_score = -1
        for item in self.catalog:
            name = item["name"].lower()
            score = 0
            for token in query.split():
                if token in name:
                    score += 1
            if score > best_score:
                best = item
                best_score = score
        return best if best_score > 0 else None

    def search(self, state: RequirementState, top_k: int = 10) -> List[Dict[str, Any]]:
        query_parts = []
        if state.role:
            query_parts.append(state.role)
        if state.seniority:
            query_parts.append(state.seniority)
        query_parts.extend(state.must_have_skills)
        if state.wants_personality:
            query_parts.append("personality behavioral fit")
        if state.wants_cognitive:
            query_parts.append("cognitive aptitude reasoning")
        if state.wants_coding:
            query_parts.append("technical coding programming")
        if state.wants_leadership:
            query_parts.append("leadership management")

        query = " ".join(query_parts).strip() or "general hiring assessment"
        q_vec = self.embedder.encode([query])[0]

        if self.catalog_embeddings is None or len(self.catalog_embeddings) == 0:
            return []

        scores = np.dot(self.catalog_embeddings, q_vec)
        rank_idx = np.argsort(-scores)

        filtered: List[Dict[str, Any]] = []
        for idx in rank_idx:
            item = self.catalog[int(idx)]
            if self._passes_filters(item, state):
                filtered.append(item)
            if len(filtered) >= top_k:
                break
        return filtered[:top_k]

    def _passes_filters(self, item: Dict[str, Any], state: RequirementState) -> bool:
        category = item.get("category", "").lower()
        skills = [skill.lower() for skill in item.get("skills", [])]
        keys = [k.lower() for k in item.get("keys", [])]
        haystack = " ".join(
            [
                item.get("name", "").lower(),
                category,
                " ".join(skills),
                " ".join(keys),
                item.get("description", "").lower(),
            ]
        )

        if state.wants_personality and "personality" not in category and "behavior" not in category:
            # allow mixed recommendation when multiple categories requested
            if not (state.wants_cognitive or state.wants_coding or state.wants_leadership):
                return False
        if state.wants_cognitive and "cognitive" not in category:
            if not (state.wants_personality or state.wants_coding or state.wants_leadership):
                return False
        if state.wants_leadership and "leadership" not in category:
            if not (state.wants_personality or state.wants_coding or state.wants_cognitive):
                return False

        if state.must_have_skills:
            skill_hits = sum(1 for skill in state.must_have_skills if skill.lower() in haystack)
            if skill_hits == 0 and state.wants_coding:
                return False
        return True

    def _extract_role(self, text: str) -> Optional[str]:
        patterns = [
            r"hire(?:\s+an|\s+a)?\s+([a-z0-9 \-\/]+)",
            r"hiring(?:\s+an|\s+a)?\s+([a-z0-9 \-\/]+)",
            r"role(?:\s+is|\s*:)?\s+([a-z0-9 \-\/]+)",
            r"need(?:\s+an|\s+a|\s+some)?\s+([a-z0-9 \-\/]+)",
            r"looking for(?:\s+an|\s+a)?\s+([a-z0-9 \-\/]+)",
            r"searching for(?:\s+an|\s+a)?\s+([a-z0-9 \-\/]+)",
            r"solution for(?:\s+an|\s+a)?\s+([a-z0-9 \-\/]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                role = re.split(r"[.,;!?]", match.group(1))[0].strip()
                if role:
                    return role
        return None

    def _extract_seniority(self, text: str) -> Optional[str]:
        for token in ["entry", "junior", "mid", "senior", "lead", "manager", "director", "vp", "executive", "cxo"]:
            if token in text:
                return token
        return None

    def _extract_skills(self, text: str) -> List[str]:
        known = [
            "java",
            "python",
            "javascript",
            "sql",
            "backend",
            "frontend",
            "leadership",
            "stakeholder communication",
            "problem solving",
            "debugging",
            "api",
        ]
        found = [skill for skill in known if skill in text]
        return list(dict.fromkeys(found))
