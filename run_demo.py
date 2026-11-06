from rag import NASAResearchRAG

if __name__ == "__main__":
    rag = NASAResearchRAG()
    result = rag.ask('What is the mission of NASA?')
    print('Answer:', result['answer'])
    print('Sources:', result['sources'])
