from dataclasses import dataclass
from database.BaseClasses import BaseKVStorage
from typing import Union, TypeVar, Optional
import os
from operators.utils import load_json_file, logger, write_json


T = TypeVar("T")
# dict[str, T], T -> Json
# 如果想明确格式可以写成class KeyStringValueJsonDatabase(BaseKVStorage[XXXFormat])
# 但是我们要做的通用容器，所以不明指，Python也不会检查，他会在父类中的Generic[T]中定义好
@dataclass
class KeyStringValueJsonDatabase(BaseKVStorage):
    
    # 因为这个类的存储是文件持久化 KV 存储，它不是用数据库，也不是存在 Redis 里
    # 把所有 key-value 对象 写入 JSON 文件中，下次启动时再从 JSON 文件加载回来。
    
    # 不要写async def __post__init__，dataclass不支持
    
    # post_init 可加入文件夹自动创建（更健壮）
    # 目前默认 working_dir 和 JSON 文件路径都已经存在，但如果路径不存在就会报错
    def __post_init__(self):
        # 这个路径决定了 KV 存储数据文件要保存在哪个目录里。
        self.working_dir = self.global_config["working_dir"]
        # working_dir下面的每个命名空间namespace就是会有一个单独的文件夹
        # 一个 KV 存储实例对应的文件名，和逻辑上是存什么类型的数据。
        # kv = XXXX(namespace="chunk", global_config={"working_dir": "/path/to/data"})
        self._file_name = os.path.join(self.working_dir, f"{self.name_space}.json")
        # 可以让你复用同一个 JsonKVStorage 逻辑，但用在多个用途（chunk、report、summary 等）。
        # 避免所有数据都存在同一个 JSON 文件里，不好管理。按照用途分文件，解耦不同数据类型的存储，结构清晰。
        self._data = load_json_file(self._file_name) or {}
        logger.info(f"Load KV Storage {self.name_space}")
        
    # 暂时没想好要做什么？比如可以清空KV数据，比如self._data = {}或者备份之前的数据？
    # 反正raise NotImplementedError，不实现也没关系
    async def index_start_callback(self):
        # print(f"Indexing started for {self.name_space} storage.")  
        pass  
    # async query_back也是，暂时不实现
    
    # 写回硬盘
    async def index_done_callback(self):
        try:
            write_json(self._data, self._file_name)
        except Exception as e:
            logger.error(f"Failed to write JSON to {self._file_name}: {e}")
        
    async def get_all_keys(self) -> list[str]:
        return list(self._data.keys())
    
    async def get_data_by_id(self, id: str) -> Union[dict, None]:
        return self._data.get(id, None)
    
    # 如果没有指明字段，即fields = None， 用set和查找比list更快
    async def get_data_by_ids(self, ids: list[str], fields: Optional[set[str]] = None) -> list[Union[dict, None]]:
        if fields is None:
            return [self._data.get(id) for id in ids]
        
        result = []
        for id in ids:
            data = self._data.get(id, None) # -> Dict
            if data is None:
                # continue 直接跳过会导致结果不对位
                result.append(None)
                continue
            # filtered_data = [data.get(field) for field in fields]
            filtered_data = {k: v for k, v in data.items() if k in fields}
            result.append(filtered_data)
            
        return result
    
    # 
    async def get_keys_to_be_inserted(self, external_keys_to_be_inserted: list[str]) -> set[str]:
        # keys_in_database = [k for k, v in self._data.items()] 这样写很慢，dict是哈希表实现的，是O(1)草查找
        keys_already_in_database = set(self._data.keys())
        # data: list[str]，不需要转成set，因为for本身就是O(1)，效率已经很高了
        # 除非你要去重，多次查找
        keys_to_be_inserted = [k for k in external_keys_to_be_inserted if k not in keys_already_in_database]
        return set(keys_to_be_inserted)
            
            
    async def update_or_insert(self, data: dict[str, T]) -> None:
        self._data.update(data) # 单条数据是O(1)，不必更改，因为底层是哈希表已经很快了
        
        # 因为插入往往是单条数据？所以不需要，如果后面有大量插入我看看
        # for k, v in data.items():
        #     if k not in self._data or self._data[k] != v:
        #         self._data[k] = v

        
    # 清空内存数据
    async def clear_all(self):
        self._data = {}      
        
    def print_all_data(self) -> None:
        """打印当前存储的所有数据"""
        if not self._data:
            logger.info(f"No data in {self.name_space} storage.")
        else:
            logger.info(f"Data in {self.name_space} storage:")
            for key, value in self._data.items():
                logger.info(f"Key: {key}, Value: {value}")