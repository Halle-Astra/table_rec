import time
import os
from utils.io import read_table_infos
from dataclasses import dataclass
import glob
from utils.table_det import TableDetector
import numpy as np
from PIL import Image
from loguru import logger
import torch
import multiprocessing

# 设置multiprocessing启动方式为spawn

# 接下来可以安全地使用multiprocessing和CUDA
from multiprocessing import Pool, Process, Queue
import tqdm

n_proc = 40


@dataclass
class tableInfo:
    file_name: str
    file_path: str
    source_code: dict


def process_func(table_info, message_queue, png_files, table_detector, process_num=None):
    # table_info = tableInfo(**table_info)
    paper_root = os.path.split(table_info.file_path)[0]
    # png_files = glob.glob(os.path.join(paper_root, 'page_*'))

    root_temp, paper_id = os.path.split(paper_root)
    discipline = os.path.split(root_temp)[-1]
    table_save_root = os.path.join('tables', discipline, paper_id)
    table_end_file = os.path.join(table_save_root, 'end.txt')

    if not os.path.exists(table_save_root):
        os.makedirs(table_save_root, exist_ok=True)

    # table_detector = TableDetector('assets/layout_models/model_final.pth', 'assets/layout_models',
    #                                process_num%8)

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
    message_queue.put('fff')

def watch_func(message_queue, end_num):
    cnt = 0
    bar = tqdm.tqdm(total=end_num)
    while cnt < end_num:
        try:
            m = message_queue.get(timeout=5)
            if m is not None and isinstance(m, str) and m == 'fff':
                cnt += 1
                bar.update(1)
        except :
            time.sleep(3)
    bar.close()
    logger.info('end of all...............')



if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

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

    tds = []
    for i in range(n_proc):
        table_detector = TableDetector(
            'assets/layout_models/model_final.pth',
            'assets/layout_models',
            i%8
        )
        tds.append(table_detector)
    # pool = Pool(100)
    message_queue = Queue(maxsize=10000000)

    context = torch.multiprocessing.get_context('spawn')
    p = context.Process(target=watch_func, args=(message_queue, len(infos)))
    p.start()
    workers = []
    for i, table_info in enumerate(infos):
        # pool.apply_async(process_func, args=(
        #     table_info,
        #     message_queue,
        #     i
        # ))
        table_info = tableInfo(**table_info)
        paper_root = os.path.split(table_info.file_path)[0]
        png_files = glob.glob(os.path.join(paper_root, 'page_*'))

        root_temp, paper_id = os.path.split(paper_root)
        discipline = os.path.split(root_temp)[-1]
        table_save_root = os.path.join('tables', discipline, paper_id)
        table_end_file = os.path.join(table_save_root, 'end.txt')
        if os.path.exists(table_end_file):
            logger.info(f'skip since {table_save_root} finished')
            message_queue.put('fff')
            continue


        p_i = context.Process(target=process_func, args=(table_info, message_queue, png_files, tds[i%n_proc], ))
        p_i.start()
        workers.append(p_i)
        if len(workers) == n_proc:
            os.system('nvidia-smi')
            for p_i in workers:
                p_i.join()
            workers = []
    # pool.close()
    # pool.join()
