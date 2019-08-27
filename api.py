# -*- coding: utf-8 -*-
from flask import Flask, jsonify, abort, make_response, request, redirect, url_for, render_template
from flask_cors import CORS
from datetime import datetime
import json
import traceback
import urllib
import fasttext
from time import time
import numpy as np
import pickle

from googletrans import Translator
translator = Translator()

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

import json

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
@app.route('/search/', methods=['GET'])
def search():
  """
  Function: search
  Summary: コード検索
  """
  # パラメータの取得
  lang, query = __get_search_param()

  # 1. 翻訳
  trans_en = translator.translate(query).text
  trans_en_word_list = trans_en.lower().split()
  print(trans_en, trans_en_word_list)

  # 2. 分散表現
  query_embed = get_word_list_vector(trans_en_word_list, model)
  
  # 3. 類似度検索
  sorted_sim_var_list = search_by_query(trans_en_word_list, model, all_var_pw_emb_list)

  # 4. リポジトリ情報取得
  top_similar_list = []
  for var_data in sorted_sim_var_list[:20]:
    var = var_data['var']
    res_dict = word_repo_dict[var]
    res_dict['repo'] = res_dict['repo_name']
    res_dict['var'] = var
    top_similar_list.append(res_dict)

  print(top_similar_list)

  return __success_response({
    "lang": lang,
    "query": query,
    "trans_en": trans_en,
    "top_similar": top_similar_list,
    # "embed": query_embed.tolist(),
  })

def get_word_list_vector(word_list, model):
    # 文ベクトルの取得
    assert type(word_list) == list
    return np.sum([model.get_word_vector(word) for word in word_list], axis=0)

# MARK: Search
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def search_by_query(query, model, all_var_pw_emb_list):
    query_vector = get_word_list_vector(query, model)
    for var_dict in all_var_pw_emb_list:
        var_dict['cos_sim'] = cos_sim(query_vector, var_dict['vector'])
    sorted_sim_var_list = sorted(all_var_pw_emb_list, key=lambda x:  x['cos_sim'], reverse=True)
    return sorted_sim_var_list


def __get_search_param():
  # クエリパラメータ
  lang = request.args.get('lang', '')
  query = request.args.get('q', '')
  return (lang, query)


def __getRequestURL():
  return urllib.parse.unquote(request.url)

# レスポンス系
def __error_response(e,_dict=None):
  _error_message = traceback.format_exc()
  logger.error(_error_message)
  try:
    __post_slack(
      _error_message,
      pretext="エラーを検出しました\n{}".format(__getRequestURL()),
      title="ERROR"
      )
  except Exception as e:
    logger.error(e)
  result = __make_message("Failed",str(e),_dict)
  return __make_json_response(result)

def __success_response(_dict=None):
  result = __make_message("OK","None",_dict)
  return __make_json_response(result)

def __make_json_response(result):
  response = make_response(jsonify(result))
  response.headers['Access-Control-Allow-Origin'] = '*'
  return response

def __make_message(state,_state_msg,_dict):
  result = {
  "ret":            state,
  "error_message":  _state_msg
  }
  if _dict is not None:
    result.update(_dict)
  return result


if __name__ == '__main__':
  global model
  st_time = time()
  print('load model')
  model = fasttext.load_model('model/wiki-news-300d-1M-subword.bin')
  print(f'model loaded in {time()-st_time}s')


  global all_var_pw_emb_list, word_repo_dict
  st_time = time()
  load = lambda path: pickle.load(open(path, mode='rb'))
  print('load data')
  all_var_pw_emb_list = load('./data/all_var_pw_emb_list.pickle')
  word_repo_dict = load('./data/word_repo_dict.pickle')
  print(f'data loaded in {time()-st_time}s')

  host = config['app']['HOST']
  port = config['app']['PORT']
  print("app starting...(host:{host},port:{port})".format(host=host,port=port))
  app.run(host=host, port=port, debug=False)