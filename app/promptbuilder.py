

# PromptBuilder is responsible for constructing the prompt that will be sent to the LLM.
# It takes the user query and the retrieved documents, formats them into a coherent prompt,
# and ensures that the LLM has all the necessary information to generate a relevant answer.
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

                    Use ONLY the provided context.

                    Recommend the book that BEST match the user's request, even if the match is partial.

                    Recommend at most 3 books. If there are more than 3 relevant books, choose the 3 that are most relevant.

                    If no books are even remotely relevant, say "I don't know".

                    Briefly explain why each recommendation matches.
                    Context:
                    {context}

                    User question:
                    {query}

                    Answer clearly and concisely:
                    """
        return prompt