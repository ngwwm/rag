import os
from openai import OpenAI

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set!")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Make a simple API call
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # You can change this to another model like "gpt-4" if you have access
        messages=[
            {"role": "user", "content": "Hello! Tell me a fun fact."}
        ],
        max_tokens=50
    )
    # Print the response
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
