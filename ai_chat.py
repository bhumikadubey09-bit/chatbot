"""AI Chatbot Engine for TitleLock Document Verification.

Supports:
1. LangChain + Google Gemini LLM (when GEMINI_API_KEY / GOOGLE_API_KEY is configured)
   - Strict grounded prompt preventing hallucination
   - Explicit distinction between document data and database records
   - Evidence-based answers with field-level citations
2. Rule-Based Deterministic Engine (functional fallback when no API key is present)
   - Answers all core land title verification questions
   - Completely offline and deterministic
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("titlelock.ai_chat")

SYSTEM_GROUNDING_PROMPT = """You are TitleLock AI, an expert land title and cadastral document verification assistant.
Your role is to analyze the user's uploaded land title document and explain the verification results against the TitleLock Cadastre Database.

GROUNDING & SAFETY RULES:
1. Grounding: Answer strictly using ONLY the provided Document Content and Authorized Database Record.
2. No Hallucinations: Do not assume, invent, or extrapolate unmentioned facts. If a detail is absent or unextracted, state clearly that it is not available.
3. Attribution: Always distinguish between what is in the uploaded document and what is in the TitleLock database (e.g., "The document specifies..." vs "In the TitleLock database, the record shows...").
4. Determinism: Do NOT alter or question the deterministic verification status. The verification status was established by database comparison. Explain the reasoning based on the field comparisons.
5. Legal Disclaimer: Always maintain that a database match confirms record alignment but does NOT constitute legal title approval or guarantee against physical disputes.
6. Strictly Read-Only: You cannot modify database records, approve title transfers, or register deeds.

CONTEXT INFORMATION:
=== EXTRACTED DOCUMENT ENTITIES ===
{document_entities}

=== RAW EXTRACTED DOCUMENT TEXT (SNIPPET) ===
{raw_text_snippet}

=== DATABASE VERIFICATION RESULT ===
Status: {verification_status}
Data Source: {data_source}
Summary: {verification_summary}
Matched Database Record: {database_record}
Field Comparisons: {comparisons_summary}
"""


class DocumentAIChatbot:
    """Answers user queries regarding uploaded documents and cadastral verification."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        ).strip()
        self._llm = None
        self._init_llm()

    def _init_llm(self) -> None:
        """Initialize LangChain Google Gemini model if API key is provided."""
        if not self.api_key:
            logger.info("No Gemini API key configured. Operating in Rule-Based Deterministic QA mode.")
            return

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.api_key,
                temperature=0.1,  # Low temperature for strict factual grounding
                max_output_tokens=1024,
            )
            logger.info("LangChain Google Gemini chatbot initialized successfully.")
        except Exception as exc:
            logger.warning("Failed to initialize Google Gemini LLM: %s. Using deterministic fallback.", exc)
            self._llm = None

    @property
    def is_llm_active(self) -> bool:
        return self._llm is not None

    @property
    def engine_name(self) -> str:
        if self._llm is not None:
            return "Google Gemini 2.0 (LangChain Grounded)"
        return "TitleLock Deterministic QA Engine (Offline Rule-Based)"

    def answer_question(
        self,
        question: str,
        verification_data: Dict[str, Any],
        raw_text: str = "",
        fields_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Answer question using Gemini if available, else use deterministic fallback."""
        clean_q = question.strip()
        if not clean_q:
            return {
                "answer": "Please ask a question regarding the uploaded document or its verification status.",
                "engine": self.engine_name,
                "is_llm": self.is_llm_active,
                "disclaimer": "Informational verification assistant only.",
            }

        if self._llm is not None:
            try:
                return self._answer_with_gemini(clean_q, verification_data, raw_text, fields_data)
            except Exception as exc:
                logger.warning("Gemini LLM call failed (%s). Falling back to deterministic QA engine.", exc)

        # Fallback to intelligent deterministic rule-based engine
        return self._answer_with_rules(clean_q, verification_data, raw_text, fields_data)

    def _answer_with_gemini(
        self,
        question: str,
        verification_data: Dict[str, Any],
        raw_text: str,
        fields_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Query Google Gemini via LangChain with grounded prompt."""
        from langchain_core.messages import HumanMessage, SystemMessage

        fields = fields_data or {}
        v = verification_data.get("verification", verification_data)
        record = v.get("record") or {}
        comparisons = v.get("comparisons") or []

        doc_entities_str = "\n".join(
            f"- {k}: {v}" for k, v in fields.items() if v is not None
        ) or "None extracted"

        comparisons_str = "\n".join(
            f"- {c.get('label', c.get('field'))}: Document='{c.get('document_value')}' | Database='{c.get('database_value')}' -> {c.get('status')} ({c.get('notes')})"
            for c in comparisons
        ) or "No field comparisons available"

        system_content = SYSTEM_GROUNDING_PROMPT.format(
            document_entities=doc_entities_str,
            raw_text_snippet=raw_text[:3000] if raw_text else "No raw text available",
            verification_status=v.get("status", "UNKNOWN"),
            data_source=v.get("source", "Unknown"),
            verification_summary=v.get("summary", "No summary"),
            database_record=str(record) if record else "No matching record located",
            comparisons_summary=comparisons_str,
        )

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=question),
        ]

        response = self._llm.invoke(messages)
        answer_text = response.content if isinstance(response.content, str) else str(response.content)

        return {
            "answer": answer_text,
            "engine": "Google Gemini 2.0 (LangChain Grounded)",
            "is_llm": True,
            "disclaimer": v.get("disclaimer", "Not proof of legal title authenticity."),
        }

    def _answer_with_rules(
        self,
        question: str,
        verification_data: Dict[str, Any],
        raw_text: str,
        fields_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Intelligent deterministic rule-based answer generator."""
        q = question.lower()
        v = verification_data.get("verification", verification_data)
        record = v.get("record") or {}
        fields = fields_data or {}
        comparisons = v.get("comparisons") or []
        status = v.get("status", "UNKNOWN")
        source = v.get("source", "Connected Data Source")

        # 1. Existence / Verification status
        if any(w in q for w in ("exist", "found", "database", "record", "match", "status")):
            if status == "RECORD FOUND":
                ans = (
                    f"Yes, this property exists in the TitleLock database ({source}). "
                    f"A matching parcel was identified for ULPIN '{record.get('ulpin')}' "
                    f"(Survey No: '{record.get('survey_number')}'). All compared fields match the registered records."
                )
            elif status == "DATA MISMATCH":
                mismatches = [c for c in comparisons if c.get("status") == "MISMATCH"]
                m_details = "; ".join(f"{m.get('label')}: Doc '{m.get('document_value')}' vs DB '{m.get('database_value')}'" for m in mismatches)
                ans = (
                    f"The property record was located for ULPIN '{record.get('ulpin')}', but a DATA MISMATCH was detected. "
                    f"Discrepancies found: {m_details}."
                )
            elif status == "RECORD NOT FOUND":
                ans = (
                    f"No matching property record exists in {source}. "
                    f"The searched identifiers (ULPIN: '{fields.get('ulpin', 'N/A')}', Survey: '{fields.get('survey_number', 'N/A')}') "
                    "do not correspond to any registered cadastre parcel."
                )
            elif status == "MULTIPLE MATCHES":
                candidates = v.get("candidate_matches", [])
                ans = (
                    f"The survey number '{fields.get('survey_number')}' matches multiple ({len(candidates)}) distinct property parcels in {source}. "
                    "Because the uploaded document lacks a unique ULPIN, automatic verification cannot confirm which parcel is referenced."
                )
            else:
                ans = f"Verification status: {status}. {v.get('summary', 'Unable to complete verification due to missing identifiers.')}"

        # 2. Owner queries
        elif "owner" in q or "titleholder" in q or "registered to" in q:
            doc_o = fields.get("current_owner") or "Not extracted from document"
            db_o = record.get("current_owner") or "No database record"
            co_owners = record.get("owners", [])
            co_str = ""
            if len(co_owners) > 1:
                co_list = [f"{o.get('name', 'Owner')} ({o.get('share_percent', 0)}%)" for o in co_owners]
                co_str = f" Co-owners registered: {', '.join(co_list)}."

            owner_comp = next((c for c in comparisons if c.get("field") == "current_owner"), None)
            comp_status = f" Status: {owner_comp.get('status')} ({owner_comp.get('notes')})." if owner_comp else ""

            ans = (
                f"Document Owner: '{doc_o}'. "
                f"Registered Database Owner: '{db_o}'.{co_str}{comp_status}"
            )

        # 3. Area / Extent queries
        elif "area" in q or "extent" in q or "size" in q or "sqm" in q:
            doc_a = fields.get("area_sqm")
            db_a = record.get("area_sqm")
            area_comp = next((c for c in comparisons if c.get("field") == "area_sqm"), None)
            comp_note = f" Status: {area_comp.get('status')} ({area_comp.get('notes')})." if area_comp else ""

            ans = (
                f"Document Area: {doc_a} sqm ({fields.get('area_raw', 'raw not specified')}). "
                f"Registered Database Area: {db_a} sqm.{comp_note}"
            )

        # 4. ULPIN / Property ID
        elif "ulpin" in q or "bhu-aadhaar" in q or "property id" in q or "parcel id" in q:
            doc_u = fields.get("ulpin") or "Not extracted"
            db_u = record.get("ulpin") or "Not found in database"
            ans = f"Document ULPIN: '{doc_u}'. Database ULPIN: '{db_u}'."

        # 5. Survey / Khasra number
        elif "survey" in q or "khasra" in q or "gat" in q or "plot" in q:
            doc_s = fields.get("survey_number") or "Not extracted"
            db_s = record.get("survey_number") or "Not found in database"
            ans = f"Document Survey/Khasra: '{doc_s}'. Database Survey/Khasra: '{db_s}'."

        # 6. Discrepancies / Mismatches
        elif any(w in q for w in ("mismatch", "difference", "discrepanc", "conflict", "differ")):
            mismatches = [c for c in comparisons if c.get("status") == "MISMATCH"]
            if mismatches:
                diff_list = "; ".join(
                    f"{m.get('label')}: Document has '{m.get('document_value')}' while Database has '{m.get('database_value')}' ({m.get('notes')})"
                    for m in mismatches
                )
                ans = f"The following discrepancies were identified: {diff_list}."
            else:
                if status == "RECORD FOUND":
                    ans = "No discrepancies detected. All extracted fields (ULPIN, Owner, Survey No, Area) align with registered database values."
                else:
                    ans = f"Verification status is {status}. No field-level mismatch occurred because no single matching record was linked."

        # 7. Missing details
        elif "missing" in q or "not found in document" in q or "incomplete" in q:
            missing = [k for k, v in fields.items() if v is None]
            if missing:
                readable = [k.replace("_", " ").title() for k in missing]
                ans = (
                    f"The following standard land title fields were NOT extracted from the uploaded document: "
                    f"{', '.join(readable)}. The system does not invent missing fields."
                )
            else:
                ans = "All primary land record fields (ULPIN, Survey Number, Owner, Area, Location, Deed details) were successfully extracted."

        # 8. Document Summary
        elif any(w in q for w in ("summar", "overview", "what is this", "explain document", "about")):
            doc_type = fields.get("document_type") or "Land Record / Property Document"
            u = fields.get("ulpin") or "Unspecified"
            s = fields.get("survey_number") or "Unspecified"
            o = fields.get("current_owner") or "Unspecified"
            a = f"{fields.get('area_sqm')} sqm" if fields.get("area_sqm") else "Unspecified"
            loc = fields.get("property_location") or "Unspecified"
            reg_no = fields.get("registration_number") or "Unspecified"

            ans = (
                f"Document Summary: This appears to be a '{doc_type}'. "
                f"Stated Owner: {o} | ULPIN: {u} | Survey No: {s} | Area: {a} | "
                f"Location: {loc} | Deed/Reg No: {reg_no}. "
                f"TitleLock verification against {source} resulted in: {status}."
            )

        # 9. Why failed / Reason
        elif any(w in q for w in ("why", "fail", "reason", "unable")):
            if status == "RECORD FOUND":
                ans = "Verification did not fail! A matching record was found and all compared fields are consistent with the cadastre."
            elif status == "DATA MISMATCH":
                mismatches = [c for c in comparisons if c.get("status") == "MISMATCH"]
                reasons = [f"{m.get('label')} mismatch: {m.get('notes')}" for m in mismatches]
                ans = f"Verification failed due to data mismatch: {'; '.join(reasons)}."
            elif status == "RECORD NOT FOUND":
                ans = f"Verification failed because no property in the cadastre matches ULPIN '{fields.get('ulpin')}' or Survey No '{fields.get('survey_number')}'."
            elif status == "MULTIPLE MATCHES":
                ans = f"Verification cannot conclude because survey number '{fields.get('survey_number')}' is shared across multiple parcels and no unique ULPIN was provided."
            else:
                ans = f"Verification could not be performed: {v.get('summary', 'Insufficient data extracted from document.')}"

        # 10. Risk status / Frozen status
        elif any(w in q for w in ("frozen", "freeze", "dispute", "risk", "encumbrance")):
            is_frozen = record.get("frozen", False)
            risk = record.get("risk_status", "UNKNOWN")
            title = record.get("title_status", "UNKNOWN")
            freeze_reason = record.get("freeze_reason") or "None"
            ans = (
                f"Database Parcel Risk & Status: Title Status = '{title}', Risk Status = '{risk}'. "
                f"Frozen = {is_frozen}" + (f" (Reason: {freeze_reason})" if is_frozen else "") + "."
            )

        # General Fallback
        else:
            ans = (
                f"I can provide evidence-based answers about: "
                f"1) Whether this property exists in the database ({status}), "
                f"2) Registered Owner name comparison ('{record.get('current_owner', 'N/A')}'), "
                f"3) Survey Number and ULPIN alignment, "
                f"4) Area/Extent comparison, "
                f"5) Missing document fields, and "
                f"6) Document summary. "
                "I cannot grant legal authenticity or alter database records."
            )

        return {
            "answer": ans,
            "engine": "TitleLock Deterministic QA Engine (Offline Rule-Based)",
            "is_llm": False,
            "disclaimer": v.get("disclaimer", "Not proof of legal title authenticity."),
        }
