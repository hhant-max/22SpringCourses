import elasticsearch
import ir_datasets
import pandas as pd

# https://pypi.org/project/elasticsearch/7.9.1/
# https://www.ulam.io/blog/text-similarity-search-using-elasticsearch-and-python/

#data import 
data_doc = ir_datasets.load('clinicaltrials/2021')
#print(data_doc.docs_count())

data_query = pd.read_csv('/workspace/22SpringCourses/IR/AS/queries_2021.tsv',sep='\t',header=None,names=['id','content'])
#print(data_query)

data_relev = pd.read_csv('/workspace/22SpringCourses/IR/AS/qrels2021.txt',sep=' ',header = None, names= ['topic','iteration','document#','relevance'])
#print(data_relev)

# doc to es 
# for doc in data_doc.docs_iter()[:2]:
#     print(doc.)
#print(data_doc.docs_cls()._fields) # ('doc_id', 'title', 'condition', 'summary', 'detailed_description', 'eligibility')

# takes a long time
for doc in data_doc.docs_iter()[:1]:
    print(doc.doc_id)

# query is automatically in es 

# first send 100 of them

# into dictionary
doc_dict = {
    ''
}


doc_in_es = []

es = elasticsearch()

elasticsearch.helpers.bulk(es, doc_in_es)

