import json
from .io import read_table_infos

if __name__ == "__main__":
    fn = '../assets/table_info.txt'
    infos = read_table_infos(fn)
    info_item = infos[0]
    pass