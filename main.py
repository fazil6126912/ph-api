# filename: main.py
# Purpose: Extract date, provider_name, doc_type, title, description from a medical PDF/JPG/PNG.
# Returns None for any field that is missing.

import os
import sys
import time
import base64
import mimetypes
from typing import Optional
from dotenv import load_dotenv

from pydantic import BaseModel, Field  # Pydantic v2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load .env file for GOOGLE_API_KEY
load_dotenv()

# Optional: Google Files API client for robust PDF handling
try:
    from google import genai
    HAS_GOOGLE_GENAI = True
except Exception:
    HAS_GOOGLE_GENAI = False


class DocMetadata(BaseModel):
    date: Optional[str] = Field(
        default=None,
        description="Document date in ISO YYYY-MM-DD if determinable; else null."
    )
    provider_name: Optional[str] = Field(
        default=None,
        description="Hospital/clinic/lab/provider name as written; else null."
    )
    doc_type: Optional[str] = Field(
        default=None,
        description="One of PRESCRIPTION, LAB_REPORT, or OTHER if determinable; else null."
    )
    title: Optional[str] = Field(
        default=None,
        description="Concise half-line title if possible; else null."
    )
    description: Optional[str] = Field(
        default=None,
        description="A factual 2–3 line summary if possible; else null."
    )


def _infer_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return "application/pdf"
        if ext in [".jpg", ".jpeg"]:
            return "image/jpeg"
        if ext == ".png":
            return "image/png"
        return "application/octet-stream"
    return mime


def _upload_with_google_genai(filepath: str, mime_type: str, api_key: str, wait: bool = True, poll_secs: int = 2):
    """Uploads file to Google GenAI API if needed (PDF or large files)."""
    client = genai.Client(api_key=api_key)
    uploaded = client.files.upload(file=filepath)
    if wait:
        while getattr(uploaded.state, "name", None) == "PROCESSING":
            time.sleep(poll_secs)
            uploaded = client.files.get(name=uploaded.name)
    return {
        "type": "media",
        "file_uri": uploaded.uri,
        "mime_type": mime_type,
    }


def _inline_base64_part(filepath: str, mime_type: str):
    """Encodes file as base64 inline part."""
    with open(filepath, "rb") as fp:
        b64 = base64.b64encode(fp.read()).decode("utf-8")
    return {
        "type": "file",
        "source_type": "base64",
        "mime_type": mime_type,
        "data": b64,
    }


def _build_media_part(filepath: str, mime_type: str, api_key: str, prefer_upload: bool) -> dict:
    """Decides whether to upload to Google GenAI or inline as base64."""
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    is_pdf = mime_type == "application/pdf"
    if HAS_GOOGLE_GENAI and (prefer_upload or is_pdf or size_mb > 18):
        try:
            return _upload_with_google_genai(filepath, mime_type, api_key)
        except Exception:
            pass
    return _inline_base64_part(filepath, mime_type)


def extract_doc_metadata(
    file_path: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    prefer_upload: bool = True,
) -> DocMetadata:
    """Main function: Extract metadata from document file."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

    model_name = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    mime_type = _infer_mime(file_path)
    media_part = _build_media_part(file_path, mime_type, api_key, prefer_upload=prefer_upload)

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )

    structured_llm = llm.with_structured_output(DocMetadata, method="json_schema")

    instructions = (
        "Task: Extract metadata from the provided medical document.\n"
        "- Return a JSON object matching the schema fields exactly.\n"
        "- If a field is not present or cannot be determined, return null for that field.\n"
        "- date: Prefer an explicit written date; normalize to ISO YYYY-MM-DD if clear; else null.\n"
        "- provider_name: Hospital/clinic/lab/provider name as written; avoid guessing; else null.\n"
        "- doc_type: Use PRESCRIPTION, LAB_REPORT, or OTHER; if unclear, null.\n"
        "- title: A concise half-line title; if not obvious, null.\n"
        "- description: A factual 2–3 line summary; if insufficient info, null.\n"
        "Avoid hallucinations; do not infer missing values beyond the document."
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": instructions},
            media_part,
        ]
    )

    result: DocMetadata = structured_llm.invoke([message])
    return result


# CLI entry point (safe for imports)
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path-to-pdf-or-image>")
        sys.exit(1)

    path = sys.argv[1]
    md = extract_doc_metadata(path)
    print(md.model_dump())
