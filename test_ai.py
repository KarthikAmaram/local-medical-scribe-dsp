from openai import OpenAI  # type: ignore

# Connecting to your local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Fake key since it's local and free
)

print("Sending a message to phi3...")

try:
    response = client.chat.completions.create(
        model="phi3",
        messages=[{"role": "user", "content": "Give me a 3-bullet-point summary of why exercise is good for the heart."}]
    )
    print("\n--- AI Response ---")
    print(response.choices[0].message.content)
    print("-------------------")
except Exception as e:
    print(f"\nAn error occurred: {e}")