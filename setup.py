from setuptools import setup, find_packages

setup(
    name="graphrag-pipeline",
    version="0.1.0",
    description="Agentic GraphRAG pipeline using LangGraph and Neo4j",
    author="Your Name",
    python_requires=">=3.10",
    package_dir={"":"src"},
    packages=find_packages(where="src"),
    install_requires=[
        "langchain>=0.2.0",
        "langgraph>=0.1.0",
        "neo4j>=5.0.0",
        "faiss-cpu>=1.7.4",
        "python-dotenv>=1.0.0",
        "ragas>=0.1.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
    ],
)
