import torch
from transformers import pipeline


class MyLLMInterface(object):
    DEFAULT_MAX_NEW_TOKENS = 200

    def __init__(self, pipe):
        self.pipe = pipe

    def build_message(self, i, sender, message, llm_settings):
        raise NotImplementedError("Implement this method")

    def build_prompt(self, messages, llm_settings):
        return '\n'.join(
            [
                self.build_message(i, m["sender"], m["message"], llm_settings)
                for i, m in enumerate(messages)
            ]
        )

    def get_answer_to(self, messages, llm_settings):
        prompt = self.build_prompt(messages, llm_settings)

        res = self.pipe(
            prompt,
            max_length=None,
            max_new_tokens=llm_settings.get("max_new_tokens", self.DEFAULT_MAX_NEW_TOKENS),
        )

        for ans in res:
            return ans["generated_text"][len(prompt):]


class LlamaInteraface(MyLLMInterface):
    START_INST = "[INST]"
    END_INST = "[/INST]"

    START_SYS = "<<SYS>>"
    END_SYS = "<</SYS>>"

    def __init__(self, model_name):
        super(LlamaInteraface, self).__init__(
            pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        )

    def build_message(self, i, sender, message, llm_settings):
        # Add sys_prompt to the first message (if it exists and is full)
        if sender == "user" and i == 0 and llm_settings.get("sys_prompt"):
            message = f"{self.START_SYS}{llm_settings['sys_prompt']}{message}{self.START_SYS}"

        if sender == "user":
            return f"{self.START_INST}{message}{self.END_INST}"
        else:
            return message


class DummyInteraface(MyLLMInterface):
    def __init__(self, model_name):
        super(DummyInteraface, self).__init__(None)

    def get_answer_to(self, messages, llm_settings):
        return "Dummy Message"


model_interfaces = {
    "meta-llama/Llama-2-7b-chat-hf": LlamaInteraface
    # "meta-llama/Llama-2-7b-chat-hf": DummyInteraface
}


def create_llm(model_name):
    return model_interfaces[model_name](model_name)
