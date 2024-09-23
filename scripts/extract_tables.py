import os
from utils.io import read_table_infos
from dataclasses import dataclass
import glob
from utils.table_det import TableDetector
import numpy as np
from PIL import Image
from loguru import logger


@dataclass
class tableInfo:
    file_name: str
    file_path: str
    source_code: dict


if __name__ == "__main__":
    if not os.path.exists('assets'):
        os.mkdir('assets')
    if not os.path.exists('tables'):
        os.mkdir('tables')

    table_infos_file = 'assets/table_info.txt'
    if not os.path.exists(table_infos_file):
        table_original_file = '/data-pfs/jd/dataset/DocGenome/table_retrive.inter.txt'
        from utils.table_explore import get_table_infos_from_file

        infos = get_table_infos_from_file(table_original_file)

        f = open(table_infos_file, 'w')
        f.write('\n'.join([str(i) for i in infos]))
        f.close()
    else:
        infos = read_table_infos(table_infos_file)

    table_detector = TableDetector('assets/layout_models/model_final.pth', 'assets/layout_models')
    for table_info in infos:
        table_info = tableInfo(**table_info)
        paper_root = os.path.split(table_info.file_path)[0]
        png_files = glob.glob(os.path.join(paper_root, 'page_*'))

        root_temp, paper_id = os.path.split(paper_root)
        discipline = os.path.split(root_temp)[-1]
        table_save_root = os.path.join('tables', discipline, paper_id)
        table_end_file = os.path.join(table_save_root, 'end.txt')
        if os.path.exists(table_end_file):
            logger.info(f'skip since {table_save_root} finished')
            continue

        if not os.path.exists(table_save_root):
            os.makedirs(table_save_root)

        end_infos = []
        for image_fp in png_files:
            image = np.asarray(Image.open(image_fp))
            tables, bboxes = table_detector(image)
            image_fn = os.path.basename(image_fp)
            for i, table in enumerate(tables):
                table_path = os.path.join(table_save_root, image_fn + '{}.png'.format(i))
                table.save(table_path)
                logger.info(f'{table_path} saved.')

                end_infos.append([table_path, str(bboxes[i])])
        f = open(table_end_file, 'w')
        f.write('path\tcoco_bboxes_xywh\n')
        end_infos = ['\t'.join(i) for i in end_infos]
        end_infos = '\n'.join(end_infos)
        f.write(end_infos)
        f.close()
