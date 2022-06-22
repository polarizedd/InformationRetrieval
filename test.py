import requests
from content_analyzer import ContentAnalyzer
from tfidf_retrieval import TfIdfRetrieval
from inverse_index_retrieval import InverseIndexRetrieval


def parse_wiki_articles(cat, lim):
    s = requests.Session()

    url = 'https://en.wikipedia.org/w/api.php'

    params = {
        'cmdir': 'desc',
        'format': 'json',
        'list': 'categorymembers',
        'action': 'query',
        'cmtitle': 'Category:' + cat,
        'cmsort': 'timestamp',
        'cmlimit': lim,
        'generator': 'images'
    }

    r = s.get(url=url, params=params)
    data = r.json()
    url_list = []
    for art in data['query']['categorymembers']:
        title = art['title'].lower()
        if any(s in title for s in ('category', 'portal')) == False:
            url_list.append(art)
    return url_list

def unranked_quality_rating(ques: list,
                   closest: list, 
                   cats: dict,
                   categories: list,
                   b = 1) -> None:
    for res in range(0, len(closest)):
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for num, _, sim in closest[res]:
            if (sim < 1) and (
                    cats[categories[res]][0] <= num <= cats[categories[res]][1]):
                tp += 1
            elif (sim < 1) and ((num > cats[categories[res]][1]) or (
                    num < cats[categories[res]][0])):
                fp += 1
            elif (sim == 1) and (num > cats[categories[res]][1]) or (
                    num < cats[categories[res]][0]):
                tn += 1
            elif (sim == 1) and (
                    cats[categories[res]][0] <= num <= cats[categories[res]][1]):
                fn += 1

        Accuracy = (tp + tn) / (tp + tn + fp + fn)
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F_1 = (1 + b ** 2) * (
                    (Precision * Recall) / (b ** 2 * Precision + Recall))
        print('===========================')
        print('query:', ' '.join(ques[res]))
        print('     Accuracy = ', Accuracy,
              '\n   Precision = ', Precision,
              '\n   Recall = ', Recall,
              '\n   F_1 = ', F_1)

def ranked_quality_rating(ques: list,
                          closest: list,
                          cats: dict,
                          categories: list) -> None:
    Mean_AVG_Precision = 0
    for res in range(0, len(closest)):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        m = cats[categories[res]][1] - cats[categories[res]][0]
        Precision_k = 0
        for num, _, sim in closest[res][:m]:
            if (sim < 1) and (
                    cats[categories[res]][0] <= num <= cats[categories[res]][1]):
                tp += 1
            elif (sim < 1) and ((num > cats[categories[res]][1]) or (
                    num < cats[categories[res]][0])):
                fp += 1
            elif (sim == 1) and (num > cats[categories[res]][1]) or (
                    num < cats[categories[res]][0]):
                tn += 1
            elif (sim == 1) and (
                    cats[categories[res]][0] <= num <= cats[categories[res]][1]):
                fn += 1
            Precision_k += tp / (tp + fp)

        AVG_Precision = 1/m * Precision_k
        Mean_AVG_Precision += AVG_Precision
        print('===========================')
        print('query:', ' '.join(ques[res]))
        print('    AVG_Precision = ', AVG_Precision)
    print('\nMean_AVG_Precision = ', Mean_AVG_Precision / len(queries))


if __name__ == '__main__':
    categories = ['Architecture',
                  'History',
                  'Literature',
                  'Astronomy',
                  'Chemistry']
    main_url = 'https://en.wikipedia.org/?curid='
    limit = 50
    urls = []
    titles = []
    cats_range = {}
    count = -1
    for category in categories:
        pages = parse_wiki_articles(category, limit)
        cats_range[category] = []
        cats_range[category].append(count + 1)
        for page in pages:
            urls.append(main_url + str(page['pageid']))
            titles.append(page['title'])
            count += 1
        cats_range[category].append(count)

    queries = ['Urban Architecture',  # Для 1 категории
               'Ancient Rome',  # Для 2 категории
               'Memoir genre',  # для 3 категории
               'Night sky',  # Для 4 категории
               'Chemistry in biology']  # Для 5 категории

    content_test = ContentAnalyzer(urls, queries)
    docs = content_test.fill_corpus()
    queries = content_test.query_filter(queries)

    # TfIdfRetrieval
    docs_tfidf = content_test.tokens_to_text(docs)
    queries_tfidf = content_test.tokens_to_text(queries)

    tfidf_test = TfIdfRetrieval(urls, docs_tfidf, queries_tfidf)
    print('=======================')
    print('TfIdf Retrieval results')
    tfidf_result = tfidf_test.tfidf()

    # InverseIndexRetrieval
    inv_test = InverseIndexRetrieval(docs, queries)
    or_result = inv_test.or_retrieval()
    print('=======================')
    print('Inverse Index Retrieval results')
    print('OR retrieval', or_result)
    and_result = inv_test.and_retrieval()
    print('AND retrieval', and_result)

    # Quality rating
    unranked_quality_rating(queries, tfidf_result, cats_range, categories)

    ranked_quality_rating(queries, tfidf_result, cats_range, categories)