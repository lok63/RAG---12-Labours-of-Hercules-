from langchain_openai import ChatOpenAI

from src.config import OPENAI_KEY
import time
class ChatOpenAIWrapper(ChatOpenAI):
    def invoke(self, *args, **kwargs):
        # Call the original invoke function from the base class (optional)
        original_output = super().invoke(*args, **kwargs)
        # Add your custom logic before returning
        return original_output

def get_chat_gpt_3_5(temperature: float = 0.0) -> ChatOpenAI:
    """
    """
    llm = ChatOpenAIWrapper(
        model="gpt-3.5-turbo-0125",
        temperature=temperature,
        openai_api_key=OPENAI_KEY,
        )
    return llm

if __name__ == '__main__':
    llm = get_chat_gpt_3_5()
    res = llm.invoke("Hi")
    print(res)