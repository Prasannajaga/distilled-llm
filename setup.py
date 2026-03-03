from setuptools import find_packages, setup


setup(
    name="distilled-llm",
    version="0.1.0",
    description="Distilled LLM training package for local and Vertex AI runs",
    python_requires=">=3.10",
    packages=find_packages(include=["scripts", "scripts.*", "Cdatasets", "Cdatasets.*", "utils", "utils.*"]),
    package_data={"scripts": ["step-train.sh"]},
    py_modules=["deploy"],
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "datasets>=2.14.0",
        "tqdm>=4.65.0",
        "transformers>=5.2.0",
        "tokenizers>=0.15.0",
        "google-cloud-storage>=2.16.0",
        "pyarrow>=23.0.1",
    ],
)
