import base64
import io
import logging
import json

from cryptography.fernet import Fernet

_key = base64.urlsafe_b64encode(
    'GZSONICYPWGZSONICYPWGZSONICYPW00'.encode()).decode()
_f = Fernet(key=_key)


def open_file(filename):
    with open(filename, 'rb') as f:
        real_data = _f.decrypt(f.read())
        logging.debug(f'读取文件完成：{filename}')
        return io.BytesIO(real_data)


def save_file(output_path, data):
    with open(output_path, 'wb') as f2:
        f2.write(_f.encrypt(data))
        logging.debug(f'保存文件完成，输出路径：{output_path}')


def open_files(filename):
    with open(filename, 'rb') as f:
        file_list = f.readlines()
        filename_list = json.loads(_f.decrypt(file_list[0]))
        logging.debug(f'文件数量：{len(file_list)}，元信息：{filename_list}')
        files = {
            k: _f.decrypt(v)
            for k, v in zip(filename_list, file_list[1:])
        }
        logging.debug(f'读取文件完成：{filename}')
        return files


def save_files(output_path, data: dict):
    with open(output_path, 'wb') as f2:
        keys = list(data.keys())
        logging.debug(f'保存多个文件，文件名列表：{keys}')
        f2.write(_f.encrypt(json.dumps(keys).encode()))
        f2.write(b'\n')
        for k in keys:
            f2.write(_f.encrypt(data[k]))
            f2.write(b'\n')
        logging.debug(f'保存文件完成，输出路径：{output_path}')


def save_checkpoint(output_path, module):
    import torch
    with open(output_path, 'wb') as f2:
        with io.BytesIO() as f1:
            torch.save(module, f1)
            f1.seek(0)
            data = f1.read()

        f2.write(_f.encrypt(data))
        logging.debug(f'保存文件完成，输出路径：{output_path}')


if __name__ == '__main__':
    logging.basicConfig()
    save_files('test.bin', {'a': b'123', 'b': b'\x11\x22'})
    print(open_files('test.bin'))
