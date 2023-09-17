from langchain.llms import ChatGLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from utils import LOG


class TranslationChainGLM:
    def __init__(self, endpoint_url: str = "http://127.0.0.1:8000", verbose: bool = True):

        # 翻译任务指令始终由 System 角色承担
        template = (
            """You are a translation expert, proficient in various languages. \n
            Translates {source_language} to {target_language}."""
        )
        system_message_prompt = PromptTemplate.from_template(template)

        # 待翻译文本由 Human 角色输入
        human_template = """{text}"""
        human_message_prompt = PromptTemplate(template=human_template, input_variables=["text"])

        # 为了翻译结果的稳定性，将 temperature 设置为 0
        llm = ChatGLM(endpoint_url=endpoint_url, max_token=8000, top_p=0.9, history=[[system_message_prompt]])

        self.chain = LLMChain(llm=llm, prompt=human_message_prompt, verbose=verbose)

    def run(self, text: str, source_language: str, target_language: str) -> (str, bool):
        result = ""
        try:
            result = self.chain.run({
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
            })
        except Exception as e:
            LOG.error(f"An error occurred during translation: {e}")
            return result, False

        return result, True