import os

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

HF_TOKEN = os.getenv("HF_TOKEN")

hf = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-0.6B",
    task="text-generation",
    # pipeline_kwargs={"max_new_tokens": 100},
    model_kwargs={"huggingfacehub_api_token": HF_TOKEN} if HF_TOKEN else {},
    
)

model = ChatHuggingFace(llm=hf)
response = model._chat_model_stream_v3('''
how it the syntax wrong? given the syntax is-' And['O'] = (return i for i in range(7))'
''').output.text
print(response)