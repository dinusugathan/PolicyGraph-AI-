import json
import os
import math
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

class PolicyRetriever:
    """LangChain-compatible Document Retriever with metadata channel filtering."""

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(base_dir, "data", "akbar_policies.json")
        self.data_path = data_path
        self.documents: List[Document] = []
        self._load_documents()

    def _load_documents(self):
        """Loads Akbar Travels policies into LangChain Document instances."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Policy file not found: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            policies = json.load(f)

        for p in policies:
            # Create a LangChain Document with rich metadata
            doc = Document(
                page_content=f"{p['title']}\n{p['content']}",
                metadata={
                    "id": p["id"],
                    "category": p["category"],
                    "channel": p["channel"],
                    "title": p["title"],
                    "raw_content": p["content"]
                }
            )
            self.documents.append(doc)

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\b\w{3,}\b", text)]

    def retrieve(self, query: str, channel: str = "ALL", top_k: int = 3) -> List[Document]:
        """
        Retrieves top_k policies matching query with channel filtering.
        Channel match logic: 'ALL' matches all policies; 'B2C' matches 'B2C' and 'ALL';
        'B2B' matches 'B2B' and 'ALL'.
        """
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return self.documents[:top_k]

        scored_docs = []
        for doc in self.documents:
            doc_channel = doc.metadata.get("channel", "ALL")
            # Apply channel filtering
            if channel != "ALL" and doc_channel != "ALL" and doc_channel != channel:
                continue

            doc_tokens = self._tokenize(doc.page_content)
            if not doc_tokens:
                continue

            # Token overlap score + title boost
            title_tokens = set(self._tokenize(doc.metadata.get("title", "")))
            overlap_content = sum(1 for t in query_tokens if t in doc_tokens)
            overlap_title = sum(2 for t in query_tokens if t in title_tokens)
            
            # Category boost
            cat_tokens = set(self._tokenize(doc.metadata.get("category", "")))
            overlap_cat = sum(1.5 for t in query_tokens if t in cat_tokens)

            score = (overlap_content + overlap_title + overlap_cat) / (len(doc_tokens) ** 0.5 + 1e-5)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]

    def get_by_id(self, policy_id: str) -> Optional[Document]:
        for doc in self.documents:
            if doc.metadata.get("id") == policy_id:
                return doc
        return None
