from sampler.ollama_sampler import OllamaSampler
from .types_eval import MessageList

if __name__ == "__main__":
    # Create a message list with a user prompt
    messages: MessageList = [
        {"role": "user", "content": "Tell me a joke about computers."}
    ]

    # Initialize the sampler
    sampler = OllamaSampler(model="llama3")

    # Call the sampler
    response = sampler(messages)

    # Print the response
    print("Ollama response:", response.response_text)
