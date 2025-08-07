query_one = [
    "this is document five", ## let's say one particular document is only retrieved against a particular query
    "this is document four",
    "this is document three",
    "this is document two",
]

query_two = [
    "this is document one",
    "this is document three",
    "this is document two",
    "this is document four",
]

query_three = [
    "this is document four",
    "this is document two",
    "this is document one",
    "this is document three",
]

rrf_scores = dict()

all_docs = [query_one, query_two, query_three]

for docs in all_docs:
    for rank, doc in enumerate(docs, start=1):
        rrf_rank = 1/(60+rank)
        rrf_scores[doc] = rrf_scores.get(doc, 0) + rrf_rank
print(f"> Unsorted: {rrf_scores}")
sorted_rrf_score = sorted(rrf_scores.items(), key=(lambda x: x[1]), reverse=True)
best_docs = [doc for doc, _ in sorted_rrf_score]
print(f"> Sorted RRF Score: {sorted_rrf_score}")
print(f"> Sorted List of Docs: {best_docs}")