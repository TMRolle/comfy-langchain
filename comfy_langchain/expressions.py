from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.schema.output_parser import StrOutputParser
from langchain.llms import TextGen
import json

class LlmPromptTemplate:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt_string": ("STRING", {
                    "multiline": True, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Tell me a joke about {topic}"
                }),
            },
            "optional": {
                "chain_input": ("LANGCHAIN_RUNNABLE", ),
            },
        }

    RETURN_TYPES = ("LANGCHAIN_RUNNABLE",)
    RETURN_NAMES = ("template",)
    FUNCTION = "prompt"
    CATEGORY = "Langchain/IO"

    def prompt(self, prompt_string, chain_input=None):
        template = PromptTemplate.from_template(prompt_string)
        if chain_input:
            return (chain_input | template, )
        else:
            return (template, )

class HuggingFacePipelineNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_id": ("STRING", {"default": "bigscience/bloom-1b7"}),
                "task": (["text2text-generation", "text-generation", "summarization"], {"default": "text-generation"}),
                "device": ("INT", {"default": -1}),
                "model_kwargs_string": ("STRING", {"default": '{"temperature": 0, "max_length": 64}', "multiline": True}),
            },
            "optional": {
                "chain_input": ("LANGCHAIN_RUNNABLE", ),
            },
        }

    RETURN_TYPES = ("LANGCHAIN_RUNNABLE",)
    RETURN_NAMES = ("llm",)
    FUNCTION = "local_pipeline"
    CATEGORY = "Langchain/LLMs"

    def local_pipeline(self, model_id, task, model_kwargs_string, device, chain_input=None):
        model_kwargs = json.loads(model_kwargs_string)
        pipeline = HuggingFacePipeline.from_model_id(model_id = model_id, task = task, device = device, model_kwargs = model_kwargs)
        if chain_input:
            return (chain_input | pipeline, )
        else:
            return (pipeline, )

class OobaboogaApiNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "textgen_url": ("STRING", {"default": "http://localhost:5000"}),
                "max_new_tokens": ("INT", {"default": 25} ),
            },
            "optional": {
                "chain_input": ("LANGCHAIN_RUNNABLE", ),
            },
        }

    RETURN_TYPES = ("LANGCHAIN_RUNNABLE",)
    RETURN_NAMES = ("llm",)
    FUNCTION = "get_textgen"
    CATEGORY = "Langchain/LLMs"

    def get_textgen(self, textgen_url, max_new_tokens, chain_input=None):
        llm = TextGen(model_url=textgen_url, max_new_tokens=max_new_tokens)
        if chain_input:
            return (chain_input | llm, )
        else:
            return (llm, )



class StringOutputParserNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chain_input": ("LANGCHAIN_RUNNABLE", ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "parse"
    CATEGORY = "Langchain/Parsers"

    def parse(self, chain_input):
        return (chain_input | StrOutputParser(), )

class JsonDict:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_data": ("STRING", {"default": "{}", "multiline": True}),
            },
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("dict",)
    FUNCTION = "json_loads"
    CATEGORY = "utils"

    def json_loads(self, json_data):
        return (json.loads(json_data), )

class GetDictAttr:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dict_data": ("DICT", ),
                "dict_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_key"
    CATEGORY = "utils"

    def get_key(self, dict_data, dict_key):
        return (str(dict_data[dict_key]),  )

class InvokeChainDictString:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chain": ("LANGCHAIN_RUNNABLE", ),
                "input_dict": ("DICT", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "invoke_string"
    CATEGORY = "Langchain"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def invoke_string(self, chain, input_dict, seed):
        string_chain = chain | StrOutputParser()
        return (string_chain.invoke(input_dict), )

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LlmPromptTemplate": LlmPromptTemplate,
    "HuggingFacePipelineNode": HuggingFacePipelineNode,
    "StringOutputParserNode": StringOutputParserNode,
    "JsonDict": JsonDict,
    "InvokeChainDictString": InvokeChainDictString,
    "OobaboogaApiNode": OobaboogaApiNode,
    "GetDictAttr": GetDictAttr,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LlmPromptTemplate": "LLM Prompt Template",
    "HuggingFacePipelineNode": "HuggingFace Local Pipeline",
    "StringOutputParserNode": "String Output Parser",
    "JsonDict": "JSON to DICT",
    "InvokeChainDictString": "Invoke Chain with Dict (string)",
    "OobaboogaApiNode": "TextGen API (Oobabooga)",
    "GetDictAttr": "Get Dictionary Attribute (string)",
}
