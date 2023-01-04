import os

file_dir = 'Z:/txj/data/十分区数据/val'

path = os.walk(file_dir)
print(path)


for signal_path in path:
    print(os.path.basename(str(signal_path[0])))
