import os
from PIL import Image
import json


#
# pdf_root = 'pdf/extracted/astro-ph'
# ids = os.listdir(pdf_root)
# paper_id = ids[0]
# paper_path = os.path.join(pdf_root, paper_id)
#
# # load layout detection result
# layout_annotation_json_file = os.path.join(paper_path, "layout_annotation.json")
#
# with open(layout_annotation_json_file, "r") as f:
#     layout_annotation = json.load(f)

def get_table_infos_from_file(table_grep_path, root='/data-pfs/jd/dataset/DocGenome/extracted'):
    table_infos = []
    with open(table_grep_path) as f:
        contents = f.read()
        contents = contents.split('\n')
        if contents[-1] == '':
            contents.pop(-1)

        # infos = []
        for i in range(len(contents)):
            c_splits = contents[i].split()
            file_info = c_splits.pop(0)
            file_name, line_num = file_info.split(':')
            file_path = os.path.join(root, file_name)

            source_code_string = '{' + ''.join(c_splits) + '}'
            source_code_dct = eval(source_code_string)

            table_infos.append(dict(
                file_name=file_name,
                file_path=file_path,
                # line_num=line_num,
                source_code=source_code_dct['source_code']
            ))
    table_infos = [str(i) for i in table_infos]
    table_infos = set(table_infos)
    table_infos = list(table_infos)
    table_infos = [eval(i) for i in table_infos]
    return table_infos

#
# def complete_table_bid(table_info):
#     ann_file = 'reading_annotation.json'
#     paper_root = os.path.split(table_info['file_path'][0])



