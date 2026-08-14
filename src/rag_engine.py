# import os
# import pandas as pd
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate

# class ResearchRAGEngine:
#     def __init__(self, db_path: str = "./chroma_paper_db_v2"):
#         self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
#         self.vector_db = Chroma(
#             persist_directory=db_path,
#             embedding_function=self.embedding_function
#         )
        
#         gemini_key = os.getenv("GEMINI_API_KEY")
#         if not gemini_key:
#             raise ValueError("GEMINI_API_KEY missing from environment variables!")

#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-2.5-flash", 
#             google_api_key=gemini_key,
#             temperature=0.3
#         )
        
#         self.prompt_template = PromptTemplate(
#             input_variables=["context", "question"],
#             template="""
#             You are an expert academic research assistant analyzing trends in machine learning.
#             Use the following retrieved abstracts and metadata to answer the user's question.

#             Context Papers:
#             {context}

#             User Question: {question}

#             Provide a concise, analytical answer. If the query asks about trends over time, explicitly mention the years and topics provided in the context.
#             """
#         )

#     # Note: Increased k to 20 to gather enough data points for a good chart
#     def query(self, user_query: str, k: int = 20):
#         results = self.vector_db.similarity_search(user_query, k=k)
        
#         context_text = ""
#         trend_data = []
        
#         # Loop through all 20 for the table and chart
#         for i, doc in enumerate(results):
#             title = doc.metadata.get('title', 'Unknown Title')
#             year = doc.metadata.get('year', 'Unknown Year')
#             topic_id = doc.metadata.get('topic_id', 'Unknown Topic')
            
#             # Only use the top 5 abstracts to build the LLM context to prevent overload
#             if i < 5:
#                 context_text += f"\n--- Paper {i+1} ---\nTitle: {title}\nYear: {year}\nTopic ID: {topic_id}\nAbstract: {doc.page_content}\n"
            
#             trend_data.append({"Year": year, "Topic": topic_id, "Title": title})
            
#         formatted_prompt = self.prompt_template.format(context=context_text, question=user_query)
#         response = self.llm.invoke(formatted_prompt)
        
#         return response.content, pd.DataFrame(trend_data)
    
# # In venv: pip install gradio pandas langchain-chroma langchain-huggingface langchain-google-genai python-dotenv