import elasticsearch
import ir_datasets
import pandas as pd

#data import 
data_doc = ir_datasets.load('clinicaltrials/2021')
print(data_doc.docs_count())

data_query = pd.read_csv('/workspace/22SpringCourses/IR/AS/queries_2021.tsv',sep='\t',header=None,names=['id','content'])
print(data_query)

data_relev = pd.read_csv('/workspace/22SpringCourses/IR/AS/qrels2021.txt',sep=' ',header = None, names= ['topic','iteration','document#','relevance'])
print(data_relev)

# doc to es 
for doc in data_doc.docs_iter()[:2]:
    