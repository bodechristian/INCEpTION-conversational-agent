import requests
import json
import codecs
import logging


class UKP_Client:
    """Super lightweight implementation of client.chat.completions.create() basic functionality towards UKP inference service
    Return value also has to satisfy:
        chat_completion.usage.prompt_tokens
        chat_completion.usage.completion_tokens
        chat_completion.choices[0].message.content
        chat_completion.choices[0].finish_reason"""

    def __init__(self):
        self.chat = Chat()


class Chat:
    def __init__(self):
        self.completions = Completions()


class Completions:
    def create(self, messages, model, tools=[], temperature=0.0, parallel_tool_calls=False):
        messages = self.join_messages(messages)
        messages = json.dumps(messages)  # used to deal with multiple quotes and newlines
        payload = f'{{"model": "{model}","prompt":{messages}, "stream":false}}'
        print(payload)
        try:
            resp = requests.post(
                'http://10.167.31.201:11434/api/generate',
                data=payload
            )
            print(resp)
            if resp.status_code == 200:
                json_response = json.loads(resp.content.decode('utf-8'))
                return ReturnObject(text=json_response['response'])
        except requests.ConnectTimeout as e:
            logger = logging.getLogger('output')
            logger.error("Connection Timed out. Make sure to be connected via VPN.")

    def join_messages(self, messages):
        return "\n".join([f"<role={m['role']}>{m['content']}"for m in messages])


class ReturnObject:
    def __init__(self, text):
        self.usage = Usage()
        self.choices = [Choice(text)]


class Choice:
    def __init__(self, text):
        self.finish_reason = "stop"
        self.message = Message(content=text)


class Message:
    def __init__(self, content):
        self.content = content


class Usage:
    def __init__(self):
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
    """
        """


if __name__ == "__main__":
    # testing required functionalities
    client = UKP_Client()
    chat_completion = client.chat.completions.create(
        messages=[
            # system prompt
            {
                "role": "system",
                "content": "answer in one sentence",

            },
            {
                "role": "user",
                "content": "why is the sky blue?",
            }
        ],
        model="llama3.2",
        temperature=0.0
    )
    print(chat_completion.usage.prompt_tokens)
    print(chat_completion.usage.completion_tokens)
    print(chat_completion.choices[0].message.content)
    print(chat_completion.choices[0].finish_reason)
