import javalang
from json import JSONEncoder
from pycparser import CParser
from c_json import to_dict
import io
import ast
import os
import tokenize
import json
from clang import cindex
import pandas as pd

cindex.Config.set_library_file('/usr/lib/x86_64-linux-gnu/libclang-6.0.so.1')


def clean(d):
    rms = []
    for k, v in d.items():
        if isinstance(v, dict) and len(v) > 0:
            clean(v)
        elif isinstance(v, list) and len(v) > 0:
            if isinstance(v[0], dict):
                for sub_v in v:
                    if isinstance(sub_v, dict):
                        clean(sub_v)
            else:
                rms.append(k)
        else:
            if v is None or not isinstance(v, str):
                rms.append(k)
    for k in rms:
        d.pop(k)
    return d


class JSEncoder(JSONEncoder):

    def __init__(self, **kwargs):
        super().__init__(indent=1, **kwargs)

    def default(self, o):
        if isinstance(o, set):
            return list(o)
        else:
            return o.__dict__


def __traverse_java(node, k=None):
    d = {}
    d['type'] = k if k else 'root'
    d['children'] = []
    #if isinstance(node, str):
    if not isinstance(node, dict):
        d['value'] = str(node)
        return d
    d['value'] = ''
    if node is None or node == False or node == True:
        node = {}
    for ck, v in node.items():
        v = [v] if not isinstance(v, list) else v
        for cv in v:
            d['children'].extend(__traverse_java(cv, ck))
    return d


def ast_java(src, cnt):
    cnt[0] += 1
    if cnt[0]%1000 ==0:
        print(cnt[0])
    try:
        passtree = javalang.parse.parse(src)
        obj = json.loads(JSEncoder().encode(passtree))
    except:
        print('jason fail')
        return {}
    return __traverse_java(obj)


def tkn_java(src, cnt):
    cnt[0] += 1
    if cnt[0]%1000 ==0:
        print(cnt[0])
    try:
        return [tkn.value for tkn in list(javalang.tokenizer.tokenize(src, ignore_errors=True))]
    except:
        return ['']  


def __traverse_cpp(node):
    d = {}
    d['value'] = str(node.displayname).lower()
    d['type'] = str(node.kind).lower()
    d['children'] = []
    for child in node.get_children():
        d['children'].append(__traverse_cpp(child))
    return d


def ast_cpp(src, cnt):
    cnt[0] += 1
    if cnt[0]%100 ==0:
        print(cnt[0])
    idx = cindex.Index.create()
    tu = idx.parse('tmp.cpp', args=['-std=c++11'],
                   unsaved_files=[('tmp.cpp', src)],  options=0)
    return __traverse_cpp(tu.cursor)


def tkn_cpp(src, cnt):
    cnt[0] += 1
    if cnt[0]%100 ==0:
        print(cnt[0])
    idx = cindex.Index.create()
    tu = idx.parse('tmp.cpp', args=['-std=c++11'],
                   unsaved_files=[('tmp.cpp', src)],  options=0)
    return [t.spelling for t in tu.get_tokens(extent=tu.cursor.extent)]


def ast_c(src, cnt):
    # parser = CParser()
    # tree = parser.parse(src)
    # return __traverse_c(to_dict(tree))
    return ast_cpp(src, cnt)


def tkn_c(src, cnt):
    # parser = CParser()
    # parser.clex.input(src)
    # tokens = []
    # while True:
    #     try:
    #         tokens.append(parser.clex.token().value)
    #     except:
    #         break
    # return tokens
    return tkn_cpp(src, cnt)


def ast_python(src):
    tree = ast.parse(src)
    return json.loads(JSEncoder().encode(tree))


def tkn_python(src):
    f = io.BytesIO(src.encode('utf-8'))
    return [tok.string for tok in tokenize.tokenize(f.readline)]


def process_csv(csv_file):
    lang = os.path.splitext(
        os.path.basename(csv_file))[0]
    lexer = locals()["tkn_"+lang]
    ast = locals()["ast_"+lang]
    df = pd.read_csv(csv_file)

    def parse_src(row):
        ast = None
        try:
            ast = ast(row['flines'])
        except:
            print('Failed to parse {}-{}', row['username'], row['file'])

    def tokenize_src(row):
        tokens = None
        try:
            tokens = lexer(row['flines'])
        except:
            print('Failed to tokenize {}-{}', row['username'], row['file'])

    df['tokens'] = df.apply(lambda row: tokenize_src(row), axis=1)
    df['ast'] = df.apply(lambda row: parse_src(row), axis=1)
