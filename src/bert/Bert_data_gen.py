from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import os
import pandas as pd
import json
import sys
tqdm.pandas()


def rm_cmts(in_, language):
    if language == 'java':
        try:
            strings = in_.split('\n')
            strings = [i.strip() for i in strings]
            return_text = ''

            multiple = False
            for i in strings:
                comment = False
                if i.startswith('/*'):
                    multiple = True
                    comment = True
                if i.endswith('*/'):
                    multiple = False
                    comment = True
                if multiple == True and i.startswith('*'):
                    comment = True
                if i.startswith('//'):
                    comment = True   
                if comment == False:
                    return_text += i+'\n'
            return return_text
        except:
            return ''
    elif language == 'py':
        try:
            strings = in_.split('\n')
            strings = [i.rstrip() for i in strings]
            return_text = ''

            multiple = False
            for i in strings:
                comment = False
                if i.startswith('\'\'\''):
                    multiple = True
                    comment = True
                if i.endswith('\'\'\''):
                    multiple = False
                    comment = True
                if multiple == True:
                    comment = True
                if i.startswith('#'):
                    comment = True   
                if comment == False:
                    return_text += i+'\n'
            return return_text
        except:
            return ' '


def rm_parse(in_):
    return_text = in_
    try:
        parse(in_)
    except:
        return_text = ' '
        
    return return_text



def clean(text, language):
    text_cmt = rm_cmts(text, language)
    text_cmt_ast = rm_parse(text_cmt)
    text_cmt_ast_spctkn = text_cmt_ast.replace('<img', '').replace('https:', '')
    return text_cmt_ast_spctkn


def lang_tokenization(row, lang):
    if lang == 'java':
        try:
            return [tkn.value for tkn in list(tokenize(row))]
        except:
            print('java_tokenization')
            return [' ']  
    if lang == 'py':
        try:
            return [tok.string for tok in tokenize(BytesIO(row.encode('utf-8')).readline)]

        except:
            print('py_tokenization')
            return [' ']  

        
if __name__ == '__main__':
    year = sys.argv[1]
    language = sys.argv[2]
    
    if language == 'java':
        from javalang.parse import parse
        from javalang.tokenizer import tokenize
    elif language == 'py':
        from ast import parse
        from tokenize import tokenize
        from io import BytesIO

    ds_all = pd.read_csv('../../GCJ/gcj'+year+'.csv')
    
    ds_all['language'] = ds_all.apply(lambda row: os.path.splitext(row['file'])[1][1:] , axis=1)
    ds = ds_all.loc[ds_all.language==language]
    print(ds.shape)
    if language == 'py':
        for i in range(ds.shape[0]):
            a = ds.flines.iloc[i]
            try:
                ds.flines.iloc[i] = a.replace('\n ', '\n')
            except:
                pass
            
    ds['flines_clean'] = ds.flines.progress_apply(lambda x: clean(x, language)) 
    print("clean samples number: "+str(ds.loc[ds['flines_clean']!=' '].shape[0]))
    ds_codebert = pd.DataFrame(columns=['repo', 'path', 'func_name', 'original_string', 'language', 'code', 'code_tokens', 'docstring', 'docstring_tokens'])
    ds_codebert['repo'] = ds['username']
    ds_codebert['path'] = ds['full_path']
    ds_codebert['func_name'] = ds['task']
    ds_codebert['language'] = language
    ds_codebert['code'] = ds['flines_clean']

    # here we use language_specific tokenizer:
#     ds_codebert['code_tokens'] = ds['flines_clean'].apply(lambda row: row.split())
    ds_codebert['code_tokens'] = ds['flines_clean'].progress_apply(lambda row: lang_tokenization(row, language))
    ds_codebert['original_string'] = ' '
    ds_codebert['docstring'] = ' '
    ds_codebert['docstring_tokens'] = ' '

    ds_codebert['sha'] = ' '
    ds_codebert['url'] = ' '
    ds_codebert['partition'] = ' '

    dict_codebert = ds_codebert.to_dict('records')
    json_path = '/home/jovyan/Source_Code_Veri/data/GCJ/MS_C2T/gcj'+year+'_'+language
    with open(json_path+'.jsonl', 'w') as f:
        for i in dict_codebert:
            f.write(json.dumps(i)+'\n')

        
