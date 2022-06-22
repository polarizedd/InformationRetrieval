from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfRetrieval:
    def __init__(self, urls, documents, queries):
        self.urls = urls
        self.queries = queries
        self.documents = documents
        self.tfidf_encoder = TfidfVectorizer()

    def tfidf(self):
        tfidf_encoded_data = self.tfidf_encoder.fit_transform(self.documents)
        tfidf_encoded_queries = self.tfidf_encoder.transform(self.queries)
        res = []
        for q_id, query in enumerate(tfidf_encoded_queries):
            query = query.todense().A1
            docs = [doc.todense().A1 for doc in tfidf_encoded_data]
            id2doc2similarity = [(doc_id, doc, cosine(query, doc)) for doc_id, doc in enumerate(docs)]
            closest = sorted(id2doc2similarity, key=lambda x: x[2],
                             reverse=False)
            print("Q: %s\nFOUND:" % self.queries[q_id])
            for closest_id, _, sim in closest[:3]:
                print(
                    "    %d\t%.2f\t%s" % (
                        closest_id, sim, self.urls[closest_id]))
            res.append(closest)
        return res
