import random
import re

from flask import Flask, render_template, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
model_name = '../fine_tuning_bgm3/best_model'
app = Flask(__name__)
# Read DataFrame from CSV (assuming it is saved as queries.csv in the data folder)
def get_dataframe(file_name):
    return pd.read_pickle(file_name)
lumi_frame=get_dataframe('./bm25/bm25_article_df_final.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/bm25')
def bm25():
    df = get_dataframe('./bm25/bm25_query_df_final.pkl')
    query_pairs = df['pair'].tolist()
    return render_template('bm25.html', query_pairs=query_pairs)

@app.route('/get_bert_query_details', methods=['POST'])
def get_bert_query_details():
    selected_model = request.json['selectedModel']
    query_pair = request.json['pair']
    global model_name

    if selected_model=='BGE-M3(Coliee Finetuned)':
        model_name = '../fine_tuning_bgm3/best_model'
    elif selected_model=='BGE-M3':
        model_name = 'BAAI/bge-m3'
    elif selected_model=='Mpnet':
        model_name = 'sentence-transformers/all-mpnet-base-v2'
    print(model_name)
    df = get_dataframe('./bge-m3/bgem3_query_df_final.pkl')
    query_details = df[df['pair'] == query_pair][['article_num', 'predicted', 'precision', 'recall', 'f2']].to_dict('records')
    query_text = df.loc[df['pair'] == query_pair, 'query'].values[0]
    return jsonify(query_text=query_text, query_details=query_details)


@app.route('/get_query_details', methods=['POST'])
def get_query_details():
    query_pair = request.json['pair']
    df = get_dataframe('./bm25/bm25_query_df_final.pkl')
    query_details = df[df['pair'] == query_pair][['article_num', 'predicted', 'precision', 'recall', 'f2']].to_dict('records')
    query_text = df.loc[df['pair'] == query_pair, 'query'].values[0]
    return jsonify(query_text=query_text, query_details=query_details)

@app.route('/get_article_ids', methods=['POST'])
def get_article_ids():
    query_pair = request.json['pair']
    df = get_dataframe('./bm25/bm25_query_df_final.pkl')
    predicted_article_ids = list(set(df[df['pair'] == query_pair]['predicted'].values[0]))
    ground_truth= list(set(df[df['pair'] == query_pair]['article_num'].values[0]))
    #merge the two lists
    article_ids = predicted_article_ids + ground_truth
    article_ids = list(set(article_ids))
    return jsonify(article_ids=article_ids)

@app.route('/get_bert_article_ids', methods=['POST'])
def get_bert_article_ids():
    query_pair = request.json['pair']
    df = get_dataframe('./bge-m3/bgem3_query_df_final.pkl')
    predicted_article_ids = list(set(df[df['pair'] == query_pair]['predicted'].values[0]))
    ground_truth= list(set(df[df['pair'] == query_pair]['article_num'].values[0]))
    #merge the two lists
    article_ids = predicted_article_ids + ground_truth
    article_ids = list(set(article_ids))
    return jsonify(article_ids=article_ids)

@app.route('/get_tokens', methods=['POST'])
def get_tokens():
    article_id = request.json['article_id']

    query_id=request.json['pair']
    df = get_dataframe('./bm25/bm25_query_df_final.pkl')
    query_details = df[df['pair'] == query_id][['query_tokens','article_tokens','common_article_tokens','term_freq_query_tokens','term_freq_article_tokens','common_commentary_tokens','term_freq_commentary_tokens']]
    query_tokens=list(set((query_details['query_tokens'].iloc[0])))
    article_df=get_dataframe('./bm25/bm25_article_df_final.pkl')
    article_tokens=list(set((article_df.loc[article_df['id'] == article_id, 'article_tokens'].values[0])))

    common_tokens=sorted(list(query_details['common_article_tokens'].iloc[0][article_id]))
    term_freq_article_tokens=query_details['term_freq_article_tokens'].iloc[0]
    term_freq_query_tokens=query_details['term_freq_query_tokens'].iloc[0]

    new_article_tokens=[]

    for k in article_tokens:
        if k.isalpha():
            new_article_tokens.append(k +' : '+ str(term_freq_article_tokens[article_id][k]))
    article_tokens=new_article_tokens

    new_query_tokens=[]
    for k in query_tokens:
        if k.isalpha():
            new_query_tokens.append(k +' : '+ str(term_freq_query_tokens[k]))
    query_tokens=new_query_tokens

    #sort query tokens
    query_tokens=sorted(query_tokens)
    article_tokens=sorted(article_tokens)


    #read idf pickle file
    idf_df = get_dataframe('./bm25/idf_values.pickle')
    #get idf_values for common tokens
    idf_values=[]
    for k in common_tokens:
        idf_values.append(k +' : '+ str(round(idf_df[k],2)))
    common_tokens=idf_values

    # commentary_tokens = article_df.loc[article_df['id'] == article_id, 'commentary_tokens'].values[0]
    # if not commentary_tokens:
    #     commentary_tokens = "No commentary available"
    # else:
    #     term_freq_commentary_tokens = query_details['term_freq_commentary_tokens'].iloc[0]
    #     commentary_tokens = sorted(list(set(commentary_tokens)))
    #     common_commentary_tokens = sorted(list(query_details['common_commentary_tokens'].iloc[0][article_id]))
    #     new_commentary_tokens=[]
    #     for k in commentary_tokens:
    #         new_commentary_tokens.append(k +' : '+ str(term_freq_commentary_tokens[article_id][k]))
    #     commentary_tokens=new_commentary_tokens
    #     idf_comm_df=get_dataframe('./bm25/idf_values_commentary.pickle')
    #     comm_idf_values=[]
    #     for k in common_commentary_tokens:
    #         comm_idf_values.append(k +' : '+ str(round(idf_df[k],2)))
    #     common_commentary_tokens=comm_idf_values
    #





    return jsonify(query_tokens=query_tokens,article_tokens=article_tokens,common_tokens=common_tokens)
@app.route('/get_article', methods=['POST'])
def get_article():
    article_id = request.json['article_id']
    df = get_dataframe('./bm25/bm25_article_df_final.pkl')
    article_text = df.loc[df['id'] == article_id, 'concated_article'].values[0]
    ##remove double spaces
    article_text = article_text.replace("  "," ")

    commentary_text = df.loc[df['id'] == article_id, 'commentary'].values[0]
    if not commentary_text:
        commentary_text = "No commentary available"
    return jsonify(article_text=article_text, commentary_text=commentary_text)

@app.route('/get_bert_article_sim_score', methods=['POST'])
def get_bert_article_score():
    article_id = request.json['article_id']
    query_pair = request.json['pair']
    df = get_dataframe('./bge-m3/bgem3_query_df_final.pkl')
    query_details = df[df['pair'] == query_pair]['all_similarity'].iloc[0]
    cosine_sim = query_details[article_id]
    return jsonify(cosine_sim=cosine_sim)

@app.route('/sentence_bert')
def sentence_bert():
    df = get_dataframe('./bge-m3/bgem3_query_df_final.pkl')
    query_pairs = df['pair'].tolist()
    return render_template('sentence_bert_new.html', query_pairs=query_pairs)


# @app.route('/get_bert_sentences', methods=['POST'])
# def get_bert_sentences():
#     query_pair = request.json['pair']
#     selected_article_id = request.json['article_id']
#     query_df = get_dataframe('./bge-m3/bgem3_query_df_final.pkl')
#     article_df = get_dataframe('./bge-m3/bgem3_article_df_final.pkl')
#     query_sentences = query_df[query_df['pair'] == query_pair]['query_sentences'].iloc[0]
#     article_sentences=article_df[article_df['id'] == selected_article_id]['article_sentences'].iloc[0]
#     ##remove double spaces from article_sentences
#     article_sentences = [x.replace("  "," ") for x in article_sentences]
#     print(article_sentences)
#     #if query_sentences has '',remove it
#     query_sentences = [x for x in query_sentences if x]
#     #strip the sentences
#     # query_sentences = [x.strip() for x in query_sentences]
#     # article_sentences = [x.strip() for x in article_sentences]
#     return jsonify(query_sentences=query_sentences, article_sentences=article_sentences)


# @app.route('/get_bert_sentence_score', methods=['POST'])
# def get_bert_sentence_score():
#     query_pair = request.json['query_pair']
#     article_id=request.json['article_id']
#     selected_query_sentence = request.json['query_sentence']
#     selected_article_sentence = request.json['article_sentence']
#     df = get_dataframe('./bge-m3/bgem3_query_df_final.pkl')
#     query_details = df[df['pair'] == query_pair]['sentence_similarity'].iloc[0]
#     article_sentences=query_details[selected_query_sentence][article_id]
#     article_sentences_new={}
#     for k,v in article_sentences.items():
#         #replace double space  with single space for k
#         k=k.replace("  "," ")
#         article_sentences_new[k]=v
#     # print(article_sentences_new)
#
#
#     cosine_sim = article_sentences_new[selected_article_sentence]
#     #find sentence with the highest similarity for the selected query sentence
#     max_sim=max(article_sentences_new.values())
#     max_sim_sentence=[k for k,v in article_sentences_new.items() if v==max_sim]
#     return jsonify(cosine_sim=cosine_sim,max_sim_sentence=max_sim_sentence,max_sim=max_sim)


def calculate_cosine_similarity(query_vector, target_vector):
    dot_product = query_vector @ target_vector
    transformed_cosine_sim = (dot_product + 1) / 2
    return transformed_cosine_sim
@app.route('/highlight_sentence', methods=['POST'])
def highlight_sentence():
    #print name of selected model
    print('selected model:',model_name)
    model=SentenceTransformer(model_name)
    data = request.json
    article_id = data['article_id']
    article_text=lumi_frame.loc[lumi_frame['id'] == article_id, 'concated_article'].values[0]
    highlighted_query = data['highlighted_query']
    substring_length = data['substring_length']

    # Split article text into substrings of the specified length
    words = re.split(r'[\t ]+', article_text)
    substrings = [' '.join(words[i:i + substring_length]) for i in range(0, len(words), substring_length)]

    ##remove double spaces
    # new_substrings = [x.replace("  "," ") for x in substrings]


    # Find the substring with the highest similarity for the selected query sentence using the model
    substrings_embeddings = [model.encode(substring) for substring in substrings]
    highlighted_query_embedding = model.encode(highlighted_query)
    similarity_scores = [float(calculate_cosine_similarity(substring_embedding, highlighted_query_embedding)) for substring_embedding in substrings_embeddings]

    # Find the substring with the highest similarity
    max_sim = max(similarity_scores)
    most_similar_substring = substrings[similarity_scores.index(max_sim)]
    print(max_sim, most_similar_substring)
    # print(substring_length)
    return jsonify(highlighted_text=most_similar_substring, similarity_score=max_sim)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001,debug=True)
