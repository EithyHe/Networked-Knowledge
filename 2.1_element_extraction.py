import asyncio
from datetime import datetime
from enum import Enum
import json
import aiofiles
from tqdm import tqdm
import os
from typing import List, Dict
import aiosqlite
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import time
import contextlib
import signal
import sys

# 添加全局变量和信号处理函数
should_exit = False

def signal_handler(signum, frame):
    global should_exit
    print("\n收到终止信号,正在安全退出...")
    should_exit = True

# 添加信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

from dotenv import load_dotenv
from gpt_helper import get_llm_response, parse_gpt_response

# 加载环境变量
load_dotenv()
openai_api_key = os.getenv('MOONSHOT_API_KEY')
print(os.getenv('MOONSHOT_API_KEY'))

# 定义枚举类型 
class PromptType(Enum):
    BASE = "base"
    FRAMEWORK = "framework" 
    EXAMPLE = "example"

class TitleType(Enum):
    TYPE1 = "一、针对性：背景与需求"
    TYPE2 = "二、创新思路：核心思路和理念"
    TYPE3 = "三、网络环境：技术运用与平台支撑"
    TYPE4 = "四、实施过程：组织方式与运行模式"
    TYPE5 = "五、保障机制：政策支持与新制度"
    TYPE6 = "六、效果：创新成效与反响"
    TYPE7 = "七、主要创新点"
    TYPE8 = "八、挑战与困境"

@contextlib.asynccontextmanager
async def get_db_connection(db_path: str, max_retries: int = 3, retry_delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            db = await aiosqlite.connect(db_path, timeout=20.0)
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=10000")
            yield db
            await db.close()
            break
        except aiosqlite.OperationalError as e:
            if attempt == max_retries - 1:
                raise
            print(f"数据库连接失败,尝试重连... ({attempt + 1}/{max_retries})")
            await asyncio.sleep(retry_delay)

class EntityExtractor:
    def __init__(self, db_path: str, batch_size: int = 10):
        self.db_path = db_path
        self.batch_size = batch_size
        self.queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = asyncio.Lock()

    def get_prompt(self, prompt_type: PromptType) -> str:
        # 保持原有的prompt定义不变
        prompts = {
            PromptType.BASE: "请提取以下文本中的关键实体：",
            PromptType.FRAMEWORK: "使用结构化方式提取以下文本中的实体：",
            PromptType.EXAMPLE: """  """
        }
        return prompts.get(prompt_type)

    async def fetch_content(self, title_type: TitleType) -> List[Dict]:
        async with get_db_connection(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            query = "SELECT post_id, post_content FROM document_chunks WHERE post_module = ?"
            async with db.execute(query, (title_type.value,)) as cursor:
                rows = await cursor.fetchall()
                print(f"从数据库中提取到{len(rows)}条数据")
                return [dict(row) for row in rows]
            
    async def write_response_to_file(response_data, filename="llm_response.txt"):
        separator = "="*50
        
        try:
            if not isinstance(response_data, str):
                response_data = str(response_data)
                
            current_time=datetime.now()
            timestamp = current_time.strftime("%Y%m%d_%H%M%S")
            filename = f"llm_response_{timestamp}.txt"
            
            async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
                await f.write(f"{separator}\n")
                await f.write("写入大模型返回的原始数据:\n")
                await f.write(f"{response_data}\n")
                await f.write(f"{separator}\n")
                
            print(f"响应数据已成功写入到 {filename}")
            print(separator)
            
        except Exception as e:
            print(f"写入文件时发生错误: {str(e)}")
            print(separator)

    async def process_item(self, item: Dict, prompt: str):
        content = f"{prompt}\n{item['post_content']}"
        
        response_data=await asyncio.wrap_future(self.executor.submit(get_llm_response, content)) 
        #await self.write_response_to_file(response_data)
        entities = await asyncio.wrap_future(self.executor.submit(parse_gpt_response, response_data))
       

        async with self.lock:
            async with get_db_connection(self.db_path) as db:
                try:
                    cursor = await db.execute("SELECT * FROM document_chunks LIMIT 1")
                    columns = [description[0] for description in cursor.description]
                    
                    if 'entity' not in columns:
                        await db.execute("ALTER TABLE document_chunks ADD COLUMN entity TEXT")
                        await db.commit()
                    
                    encode_entities = json.dumps(entities, ensure_ascii=False).encode('utf-8-sig')
                    await db.execute(
                        "UPDATE document_chunks SET entity = ? WHERE post_id = ?",
                        (encode_entities, item['post_id'])
                    )
                    await db.commit()
                except aiosqlite.OperationalError as e:
                    print(f"数据库操作错误: {e}")
                    await asyncio.sleep(1)
                    raise

    async def process_batch(self):
        global should_exit
        total_items = self.queue.qsize()
        self.progress_bar = tqdm(total=total_items, desc="处理进度")
        
        while not self.queue.empty():
            if should_exit:
                print("正在终止进程...")
                break
                
            batch = []
            for _ in range(min(self.batch_size, self.queue.qsize())):
                if not self.queue.empty():
                    batch.append(self.queue.get())
            
            if batch:
                tasks = [self.process_item(item, item['prompt']) for item in batch]
                try:
                    await asyncio.gather(*tasks)
                except Exception as e:
                    print(f"批处理出错: {e}")
                    should_exit = True
                    break
                self.progress_bar.update(len(batch))
        
        self.progress_bar.close()

    async def run(self, prompt_type: PromptType, title_type: TitleType):
        prompt = self.get_prompt(prompt_type)
        contents = await self.fetch_content(title_type)
        
        for content in contents:
            content['prompt'] = prompt
            self.queue.put(content)
        
        await self.process_batch()
   
async def main():
    db_path="E:\数据处理\项目\问题导向网络化知识\\00 原始语料\\test_origin.db"
    extractor = EntityExtractor(db_path, batch_size=10)
    try:
        await extractor.run(
            prompt_type=PromptType.EXAMPLE,
            title_type=TitleType.TYPE1
        )
    except Exception as e:
        print(f"程序执行出错: {e}")
        sys.exit(1)
    finally:
        extractor.executor.shutdown()
        if should_exit:
            sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
