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



if __name__ == '__main__':
    year = sys.argv[1]
    language = sys.argv[2]
    if language == 'java':
        from javalang.parse import parse
    elif language == 'py':
        from ast import parse
    
    json_path = '/home/jovyan/Source_Code_Veri/data/GCJ/MS_C2T/gcj'+year+'_'+language
    ds_all = pd.read_csv('/home/jovyan/Source_Code_Veri/data/GCJ/gcj'+year+'.csv')
    
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
    print(ds.loc[ds['flines_clean']!=' '].shape)
    ds_codebert = pd.DataFrame(columns=['repo', 'path', 'func_name', 'original_string', 'language', 'code', 'code_tokens', 'docstring', 'docstring_tokens'])
    ds_codebert['repo'] = ds['username']
    ds_codebert['path'] = ds['full_path']
    ds_codebert['func_name'] = ds['task']
    ds_codebert['language'] = language
    ds_codebert['code'] = ds['flines_clean']
    ds_codebert['code_tokens'] = ds['flines_clean'].apply(lambda row: row.split())
    ds_codebert['original_string'] = ' '
    ds_codebert['docstring'] = ' '
    ds_codebert['docstring_tokens'] = ' '

    ds_codebert['sha'] = ' '
    ds_codebert['url'] = ' '
    ds_codebert['partition'] = ' '

    dict_codebert = ds_codebert.to_dict('records')
    with open(json_path+'.jsonl', 'w') as f:
        for i in dict_codebert:
            f.write(json.dumps(i)+'\n')

        
        
# tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")        
# ds_codebert['code_tokens'] = ds['flines_clean'+str(i)].apply(lambda row: tokenizer.encode(row))
# ds_codebert['code_tokens'] = ds_codebert.code_tokens.apply(lambda row: tokenizer.convert_ids_to_tokens(row))