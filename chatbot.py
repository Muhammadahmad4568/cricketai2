import os
from google import genai

# ============================================================
# GEMINI CRICKET COACH
# ============================================================
# CHANGES FROM YOUR ORIGINAL chatbot.py:
#
#   1. SECURITY: the API key was hardcoded in plaintext in this
#      file. That key has now been shared in a chat upload, so
#      treat it as compromised regardless of anything below —
#      regenerate/revoke it in Google's console today. Going
#      forward, the key is read from the GEMINI_API_KEY
#      environment variable (or Streamlit secrets — see below)
#      and is never written into source.
#
#   2. BUG FIX: your app.py already calls
#         ask_cricket_coach(question, context)
#      with two arguments, but this file only defined
#         ask_cricket_coach(question)
#      — a guaranteed crash the first time the chatbot page ran.
#      ask_cricket_coach now accepts an optional analysis_context
#      argument. When app.py passes the CURRENT video's real
#      technique/shot/sync data, the coach answers from that
#      instead of the old hardcoded ANALYSIS_CONTEXT example data.
#
#   3. STREAMLIT CLOUD FIX: on Streamlit Cloud, secrets are set
#      through the app's "Secrets" panel (TOML format), not a
#      shell environment variable — `export`/`setx` only works
#      locally. This now also checks st.secrets as a fallback, so
#      the same code works whether the key was set as a real env
#      var (local dev) or as a Streamlit Cloud secret (deployed).
# ============================================================


def _get_api_key():
    """Checks a real environment variable first, then falls back to
    Streamlit's secrets manager (how keys are actually set on
    Streamlit Cloud). Never hardcoded."""

    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        # st.secrets raises if no secrets.toml exists at all (e.g. running
        # this file outside Streamlit, like the CLI test mode below) —
        # that's fine, just means no key is available from that source.
        return None


API_KEY = _get_api_key()

_client = None


def _get_client():
    global _client

    if _client is not None:
        return _client

    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Set it as an environment variable "
            "or in Streamlit secrets before using the AI Cricket Coach."
        )

    _client = genai.Client(api_key=API_KEY)
    return _client


# Used only when no real analysis is available yet (e.g. before
# the user has uploaded and analyzed a video).
FALLBACK_CONTEXT = """
No batting video has been analyzed yet in this session.
There is no shot, technique, or ball-detection data available.
If asked about specific numbers, say that no analysis is
available yet and the user should upload and analyze a video first.
"""


def ask_cricket_coach(question, analysis_context=None):
    """
    Answers a cricket-coaching question using the CURRENT session's
    real analysis data (technique scores, shot data, ball-shot sync)
    when provided via analysis_context. Falls back to a generic
    "no data yet" context if none is passed.
    """

    context = analysis_context if analysis_context else FALLBACK_CONTEXT

    prompt = f"""
You are CricketAI, an AI cricket batting coach.

Your job is to explain the player's batting analysis
in a simple, helpful and understandable way.

IMPORTANT RULES:

1. Use the player's actual analysis data below.
2. Do NOT invent measurements that aren't in the data.
3. If data is unavailable for something asked, clearly say it
   is unavailable rather than guessing.
4. Explain technical cricket terms in simple language.
5. Give practical batting advice.
6. If the user asks about a particular shot, use the
   available shot information for that shot number.
7. Be encouraging but honest.
8. Do not claim that the analysis is perfect.
9. Mention that computer-vision measurements can sometimes
   contain detection errors, where relevant.

PLAYER ANALYSIS (current video):
{context}

USER QUESTION:
{question}

Answer as an AI cricket coach.
"""

    client = _get_client()

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text


# ============================================================
# CLI TEST MODE (unchanged behavior, still works standalone)
# ============================================================

def main():

    print()
    print("=" * 55)
    print("CRICKET AI - GEMINI COACH (CLI test mode)")
    print("=" * 55)
    print()
    print("Chatbot is ready! Type 'exit' to close.")
    print()

    while True:

        question = input("You: ")

        if question.lower().strip() == "exit":
            print()
            print("CricketAI: Good luck with your batting!")
            break

        if not question.strip():
            continue

        try:
            answer = ask_cricket_coach(question)
            print()
            print("CricketAI:")
            print(answer)
            print()

        except Exception as e:
            print()
            print("Gemini API Error:")
            print(e)
            print()


if __name__ == "__main__":
    main()