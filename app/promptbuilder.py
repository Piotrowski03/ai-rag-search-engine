

class PromptBuilder:
    def __init__(self):
        pass

    def build_prompt(self, query, docs):
        
        context = "\n\n".join([
            f"Title: {d[0]}\nDescription: {d[1]}\n"
            for d in docs
            ])
        

        prompt =f"""
                    You are a helpful assistant that recommends books.
                    
                    You must not hallucinate. If context is insufficient, say 'I don't know'.

                    Use ONLY the context below.

                    Context:
                    {context}

                    User question:
                    {query}

                    Answer clearly and concisely:
                    """
        return prompt