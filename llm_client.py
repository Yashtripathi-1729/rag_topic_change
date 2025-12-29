import os
import logging
from openai import OpenAI

# ------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

# ------------------------------------------------------------------
# OpenAI Client Initialization
# ------------------------------------------------------------------
try:
    OPENAI_API_KEY = "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"
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully")

except Exception:
    logger.exception("Failed to initialize OpenAI client")
    raise


# ------------------------------------------------------------------
# LLM Wrapper
# ------------------------------------------------------------------
def llm(prompt: str) -> str:
    """
    Calls OpenAI Chat Completion API.
    """
    logger.info("LLM call initiated")
    logger.debug("Prompt length: %d characters", len(prompt))

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        content = response.choices[0].message.content.strip()
        logger.info("LLM response received successfully")
        logger.debug("Response length: %d characters", len(content))

        return content

    except Exception:
        logger.exception("LLM call failed")
        raise
