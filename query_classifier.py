# query_classifier.py

AGGREGATION_KEYWORDS = [
    "total", "sum", "average", "avg", "count", "how many",
    "min", "max", "aggregate", "group by", "statistics",
    "overall", "in total", "total amount"
]

def classify_query(query: str) -> str:
    q = query.lower()

    # detect aggregation intent
    if any(word in q for word in AGGREGATION_KEYWORDS):
        return "aggregation"

    # default: semantic
    return "semantic"
