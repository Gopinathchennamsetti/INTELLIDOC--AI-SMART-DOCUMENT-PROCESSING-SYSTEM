import os
import logging
import nltk
import re

from google import genai
from google.api_core.exceptions import GoogleAPIError, RetryError, InvalidArgument

# ---------- NLTK setup ----------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------- Gemini client ----------

os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Please export it before running the app.")

client = genai.Client(api_key=API_KEY)
DEFAULT_MODEL = "gemini-2.5-flash"  # you can change this if desired


def split_text_into_chunks(text: str, max_length: int = 2000):
    """
    Splits text into chunks of a maximum length, preserving sentence boundaries.
    """
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_length:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def _call_gemini(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
) -> str:
    """
    Single helper to call Gemini and return plain text.
    """
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        )
        text = (resp.text or "").strip()
        logging.info(f"_call_gemini returned {len(text)} chars")
        return text
    except (GoogleAPIError, RetryError, InvalidArgument) as e:
        logging.error(f"Gemini API error: {e}")
        return ""
    except Exception as e:
        logging.error(f"Unexpected Gemini error: {e}")
        return ""


def identify_duplicates(text: str, model: str = DEFAULT_MODEL) -> str:
    """
    Identifies duplicate or semantically similar sentences in the text using Gemini.
    """
    chunks = split_text_into_chunks(text)
    duplicates = []
    logging.info(f'Identify duplicates: {len(chunks)} chunks')

    system_prompt = (
        "Identify and list all duplicate, repeated, and semantically similar sentences "
        "in the following text. Group them under headings like 'Exact duplicates' and "
        "'Semantic similarity'. Return a clear, human-readable list."
    )

    for chunk in chunks:
        prompt = f"{system_prompt}\n\nText:\n{chunk}"
        result = _call_gemini(prompt, model=model, temperature=0.1)
        if result and result.strip():
            duplicates.append(result.strip())

    if not duplicates:
        logging.warning("identify_duplicates: Gemini returned no content.")
        return ""

    logging.info("Duplicates identification completed")
    return "\n\n".join(duplicates)


def deduplicate_text(text: str, model: str = DEFAULT_MODEL) -> str:
    """
    Removes duplicate or semantically similar content from the text using Gemini.

    If, for some reason, all chunks fail or return empty, it falls back to the original text
    so that consolidated output is never empty.
    """
    chunks = split_text_into_chunks(text)
    deduplicated = []
    logging.info(f"deduplicate_text: {len(chunks)} chunks")

    system_prompt = (
        "You are a helpful editor. Read the following text and produce a cleaned version "
        "with duplicate or semantically very similar sentences removed. "
        "Keep all important information and unique insights. "
        "Do NOT remove entire sections unless they are clearly repeated. "
        "Return only the cleaned text, in the same language as the input."
    )

    for chunk in chunks:
        prompt = f"{system_prompt}\n\nText:\n{chunk}"
        result = _call_gemini(prompt, model=model, temperature=0.2)
        if result and result.strip():
            deduplicated.append(result.strip())

    # If everything failed or produced empty, fallback to original text
    if not deduplicated:
        logging.warning("deduplicate_text: Gemini returned no content; falling back to original text.")
        return text

    return "\n\n".join(deduplicated)


def summarize_text(text: str, model: str = DEFAULT_MODEL) -> str:
    """
    Summarizes the text using Gemini, capturing essential information and insights.
    """
    chunks = split_text_into_chunks(text)
    summaries = []
    logging.info(f"summarize_text: {len(chunks)} chunks")

    system_prompt = (
        "Distill the following text into a comprehensive summary that captures all "
        "essential information and insights. Cover key themes, findings, and relevant "
        "details. Aim for a concise, coherent summary that reflects the main points "
        "while maintaining context."
    )

    for chunk in chunks:
        prompt = f"{system_prompt}\n\nText:\n{chunk}"
        result = _call_gemini(prompt, model=model, temperature=0.2)
        if result and result.strip():
            summaries.append(result.strip())

    if not summaries:
        logging.warning("summarize_text: Gemini returned no content; falling back to original text.")
        return text

    return "\n\n".join(summaries)


# # Optional quick local test (run: python openai_utils.py)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     sample = (
#         "This is a sentence. This is a sentence repeated. "
#         "Here is another unique sentence. Here is another unique sentence."
#     )
#     print("=== DEDUPLICATED TEST ===")

#     print(deduplicate_text(sample))
