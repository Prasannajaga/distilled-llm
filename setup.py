from setuptools import find_packages, setup

setup(
    name="distilled_llm",
    version="0.1",
    packages=find_packages(),  
    py_modules=["text_vertex", "deploy"], # Explicitly include top-level un-foldered scripts
    description="My distributed LLM training package",
)
