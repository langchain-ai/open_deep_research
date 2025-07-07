from setuptools import find_packages, setup

setup(
    name="open_deep_research",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "crawl4ai",
        "tavily-python",
        "langchain",
        "langgraph",
        "langchain-groq",
        "pandas",
        "openpyxl",
        "chromadb",
        "pydantic",
        "playwright",
    ],
)
