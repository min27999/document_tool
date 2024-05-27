import pickle
import time
import sys
import document_tool as dt

doc2vec_model = dt.get_doc2vec_model('./Output/doc2vec_model_200')
logistic_model = dt.get_logistic_model('./Output/logistic_regression_model.pickle')

# argument length check
if len(sys.argv) < 2:
    print('usage: python3 main.py file.txt')
    sys.exit()

print('-' * 50)
print('텍스트 분석을 이용한 문서 이해 및 유사 정보 제공')
print('-' * 50, end='\n\n')

with open(f'./{sys.argv[1]}', 'r') as fr:
    document = fr.readlines()

if type(document) == list:
    document = ''.join(document)

# Find Category
print('Finding Category')
start_time = time.time()

category = dt.get_category(document, doc2vec_model, logistic_model)

print(f'Category: {category}, spend time: {time.time() - start_time}')

with open('./Output/tokenized_data.pickle', 'rb') as fr:
    guide_documents = pickle.load(fr)



# Find Similar Document
print('Finding Similar Document')
start_time = time.time()

similar_documents = dt.find_similar_documents(category, document, guide_documents, doc2vec_model, 3)

print(f'Similar Documents, spend time: {time.time() - start_time}')

# idx = 1
# for similar in similar_documents:
#     with open(f'similar{idx}.txt', 'w') as fr:
#         fr.writelines(similar[1])
#     idx += 1


print('Summarizing')
start_time = time.time()

summary = dt.summarize(document)

print(f'Summary, spend time: {time.time() - start_time}')

# with open('summary.txt', 'w') as fw:
#     fw.writelines(summary)

# for s in summary:
#     # print(s)
#     with open(f'summary{idx}.txt', 'w') as fr:
#         fr.writelines(s)

# WordCloud
with open('./Output/document_word.pickle', 'rb') as fr:
    document_word = pickle.load(fr)

print('Making WordCloud')
filename = input('Enter filename: ')
start_time = time.time()

tf_idf = dt.get_TF_IDF_values(document, document_word)

dt.make_wordcloud(tf_idf, filename)

with open('./Output/result.txt', 'w') as fr:
    fr.write(f'category: {category}\n\n')

    fr.write('Summary\n')
    for s in summary:
        fr.write(f'{s}\n')
    # fr.writelines(summary)
    fr.write('\nSimilar Documents\n')
    
    for similarity, document in similar_documents:
        fr.write(f'similarity: {similarity}\n')
        fr.writelines(document)
        fr.write('\n')
    # fr.writelines(similar_documents)