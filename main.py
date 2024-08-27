import os
import cohere
import ssl
import hnswlib
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title

ssl._create_default_https_context = ssl._create_unverified_context

co = cohere.Client(os.getenv("COHERE_API_KEY"))

raw_documents = [
    {
        "title": "OCPP 1.6",
        "url": "https://gvzdv.github.io/ocpp1.6/"},

    {
        "title": "The WebSocket API (WebSockets)",
        "url": "https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API"
    },
    {
        "title": "Writing WebSocket client applications",
        "url": "https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API/Writing_WebSocket_client_applications"
    }
]


class Vectorstore:
    def __init__(self, raw_documents: List[Dict[str, str]]):
        self.raw_documents = raw_documents
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.load_and_chunk()
        self.embed()
        self.index()

    def load_and_chunk(self) -> None:
        """
        Loads the text from the sources and chunks the HTML content.
        """
        print("Loading documents...")

        for raw_document in self.raw_documents:
            elements = partition_html(url=raw_document["url"])
            chunks = chunk_by_title(elements)
            for chunk in chunks:
                self.docs.append(
                    {
                        "title": raw_document["title"],
                        "text": str(chunk),
                        "url": raw_document["url"],
                    }
                )

    def embed(self) -> None:
        """
        Embeds the document chunks using the Cohere API.
        """
        print("Embedding document chunks...")

        batch_size = 90
        self.docs_len = len(self.docs)
        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i: min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            docs_embs_batch = co.embed(
                texts=texts, model="embed-english-v3.0", input_type="search_document"
            ).embeddings
            self.docs_embs.extend(docs_embs_batch)

    def index(self) -> None:
        """
        Indexes the documents for efficient retrieval.
        """
        print("Indexing documents...")

        self.idx = hnswlib.Index(space="ip", dim=1024)
        self.idx.init_index(max_elements=self.docs_len, ef_construction=512, M=64)
        self.idx.add_items(self.docs_embs, list(range(len(self.docs_embs))))

        print(f"Indexing complete with {self.idx.get_current_count()} documents.")

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieves document chunks based on the given query.

        Parameters:
        query (str): The query to retrieve document chunks for.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved document chunks, with 'title', 'text', and 'url' keys.
        """

        # Dense retrieval
        query_emb = co.embed(
            texts=[query], model="embed-english-v3.0", input_type="search_query"
        ).embeddings

        doc_ids = self.idx.knn_query(query_emb, k=self.retrieve_top_k)[0][0]

        # Reranking
        rank_fields = ["title", "text"]  # We'll use the title and text fields for reranking

        docs_to_rerank = [self.docs[doc_id] for doc_id in doc_ids]

        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v3.0",
            rank_fields=rank_fields
        )

        doc_ids_reranked = [doc_ids[result.index] for result in rerank_results.results]

        docs_retrieved = []
        for doc_id in doc_ids_reranked:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    "url": self.docs[doc_id]["url"],
                }
            )

        return docs_retrieved


vectorstore = Vectorstore(raw_documents)


def run_chatbot(message, chat_history=[]):
    # Generate search queries, if any
    response = co.chat(message=message,
                       model="command-r-plus",
                       search_queries_only=True,
                       chat_history=chat_history)

    search_queries = []
    for query in response.search_queries:
        search_queries.append(query.text)

    # If there are search queries, retrieve the documents
    if search_queries:
        print("Retrieving information...", end="")

        # Retrieve document chunks for each query
        documents = []
        for query in search_queries:
            documents.extend(vectorstore.retrieve(query))

        # Use document chunks to respond
        response = co.chat_stream(
            message=message,
            model="command-r-plus",
            documents=documents,
            chat_history=chat_history,
        )

    else:
        response = co.chat_stream(
            message=message,
            model="command-r-plus",
            chat_history=chat_history,
        )

    # Print the chatbot response and citations
    chatbot_response = ""
    print("\nChatbot:")

    for event in response:
        if event.event_type == "text-generation":
            print(event.text, end="")
            chatbot_response += event.text
        if event.event_type == "stream-end":
            if event.response.citations:
                print("\n\nCITATIONS:")
                for citation in event.response.citations:
                    print(citation)
            if event.response.documents:
                print("\nCITED DOCUMENTS:")
                for document in event.response.documents:
                    print(document)
            # Update the chat history for the next turn
            chat_history = event.response.chat_history

    return chat_history


chat_history = run_chatbot('''
In OCPP 1.6, do I need transactionId in Charging Profile?
''')
