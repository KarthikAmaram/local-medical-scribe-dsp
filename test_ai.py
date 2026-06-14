from openai import OpenAI  # type: ignore

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def fix_transcript_typos(raw_transcript):
    system_prompt = (
        "You are a strict text-cleaning tool for medical transcriptions. Your only job is to fix spelling mistakes, "
        "acoustic typos, and severe multi-word phonetic hallucinations where a microphone splits or mangles medical terms "
        "into nonsensical real English words (e.g., 'pal transitions' -> 'palpitations', 'koff' -> 'cough').\n"
        "CRITICAL PROTOCOLS:\n"
        "1. Use the surrounding sentence context to infer the true medical word if a phrase makes no sense but sounds phonetically similar.\n"
        "2. Do NOT rewrite correct narrative descriptions. Leave phrases like 'driving at work' or 'making breakfast' exactly as they are.\n"
        "3. Output ONLY the finalized text without introductions, explanations, or notes."
    )
    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_transcript}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    test_string = "The patient says they have bad numness and tingling in their right hand, especially in the thumb and index finger. It wakes them up at night and they feel like they have a weak weak grip when trying to hold a cup. They deny any shoulder pain or neck stifness while driving to work."
    cleaned_result = fix_transcript_typos(test_string)
    print("\n--- Cleaned AI Output ---")
    print(cleaned_result)
    print("-------------------------")