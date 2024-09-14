def read_table_infos(file):
    f = open(file)
    t = f.read().split('\n')
    f.close()
    t = [eval(i) for i in t if i]
    return t