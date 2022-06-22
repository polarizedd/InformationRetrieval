from collections import defaultdict


class InverseIndexRetrieval:
    def __init__(self, documents, queries):
        self.documents = documents
        self.queries = queries
        self.inverted_index = defaultdict(set)
        self.uniq = set()

    def generate_inverted_index(self):
        for doc_id, c in enumerate(self.documents):
            for term in c:
                self.inverted_index[term].add(doc_id)
        return self.inverted_index

    def or_retrieval(self):
        result = {}
        self.generate_inverted_index()
        for query in self.queries:
            for word in query:
                matches = self.inverted_index.get(word)
                if matches:
                    result[word] = matches
        return result

    def and_retrieval(self):
        result = {}
        matches_prev = set()
        self.generate_inverted_index()
        for q in range(0, len(self.queries)):
            for word in self.queries[q]:
                matches = self.inverted_index.get(word)
                try:
                    if matches & matches_prev:
                        result[' '.join(self.queries[q])] = matches & matches_prev
                    matches_prev = matches
                except TypeError:
                    result[' '.join(self.queries[q])] = 0
        return result
