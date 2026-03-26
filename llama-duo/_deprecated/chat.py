import requests
import sys

def bold(text):
    return "\033[1m" + text + "\033[0m"

# rudimentary chat app for testing
def chat(url):
    headers = {'Content-Type': 'application/json'}

    history = []

    url = f'{url}/messages'

    while True:
        user_input = input(bold("You: "))
        if user_input.lower() in [':quit', ':q', ':exit']:
            print("Exiting chat")
            break

        if user_input.lower() in [':new']:
            print("Starting new chat")
            history = []
            continue

        history.append({"role": "user", "content": user_input})

        data = {
            "system": "You are a helpful, respectful and honest assistant.",
            "max_tokens": 2048,
            "messages": history
        }

        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            assistant_message = response.json()['content']['text']
            print(bold("AI: "), assistant_message)
            history.append({"role": "assistant", "content": assistant_message})
        else:
            print("Failed to get response:", response.text)
            break

if __name__ == "__main__":
    url = "http://localhost:5555"
    if len(sys.argv) > 1:
        url = sys.argv[1]
    chat(url)
