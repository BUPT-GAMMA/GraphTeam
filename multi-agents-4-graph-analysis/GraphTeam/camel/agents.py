import json
import base64
import subprocess
from typing import Any, Dict, List, Optional
from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential
import os
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import JSONNodeParser
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from memory.get_result_from_memory import parse_raw_text_to_dict_memory, find_detailed_info_by_ids, select_by_accuracy
from camel.configs import ChatGPTConfig
from camel.rag_configs import RAGConfig
from camel.memory_configs import MemoryConfig
from camel.model_backend import ModelBackend, ModelFactory
from llama_index.core import Document  # 确保导入正确的 Document 类
import re

try:
    from openai.types.chat import ChatCompletion

    openai_new_api = True  # new openai api version
except ImportError:
    openai_new_api = False  # old openai api version

class Agent():
    r"""
    Args:
        system_message (SystemMessage): The system message for the chat agent.
        with_memory(bool): The memory setting of the chat agent.
        model (ModelType, optional): The LLM model to use for generating
            responses. (default :obj:`ModelType.GPT_3_5_TURBO`)
        model_config (Any, optional): Configuration options for the LLM model.
            (default: :obj:`None`)
        memory_config (Any, optional): Configuration options for the Memory model.
    """

    def __init__(
            self,
 
            model: Optional[str] = "gpt-4o-mini",
            model_config: Optional[Any] = None,
            rag_config: Optional[Any] = None,
            memory_config: Optional[Any] = None,
    ) -> None:

        self.model_name: str = (model if model is not None else "gpt-4o-mini")
        self.model_config: ChatGPTConfig = model_config or ChatGPTConfig()
        self.rag_config: RAGConfig = rag_config or RAGConfig()
        self.memory_config: MemoryConfig = memory_config or MemoryConfig()
        self.model_backend: ModelBackend = ModelFactory.create(self.model_name, self.model_config.__dict__)


    @retry(wait=wait_exponential(min=5, max=60), stop=stop_after_attempt(5))
    def generate_response(self, messages) -> ChatCompletion:
        pass

def extract_code(text):
    text = str(text)
    matches = re.findall(r"```[ \t]*[Pp]ython[ \t]*(.*?)```", text, re.DOTALL)
    filtered_matches = [match for match in matches if not re.search(r"\bpip install\b", match)]
    code = '\n'.join(filtered_matches)

    if code == '':
        return text
    else:
        return code
    

class DemandAnalysisAgent(Agent):

    def __init__(
            self,
            model: Optional[str] = "gpt-4o-mini",
            model_config: Optional[Any] = None,
    ) -> None:
        super().__init__(model, model_config)

    def generate_response(self, messages, phase_env) -> json:
        if phase_env['model'] != "Default" and phase_env['model'] != self.model_name:
            self.model_name = phase_env['model']
            self.model_backend = ModelFactory.create(self.model_name, self.model_config.__dict__)
        content = self.model_backend.run(messages=messages,response_format={"type": "json_object"}).choices[0].message.content.strip()
        try:
            if content.startswith("```json\n"):
                content = content.split("```json\n")[1].split("\n```")[0]
            json_response = json.loads(content)
            return json_response, phase_env
        except:
            # 如果没有抓取到format，记录调试信息，但不抛出异常
            print(f"Warning: Content does not contain a valid format. Full content: {content}")
            # 返回完整的内容或继续处理，允许下一个任务继续执行
            return content, phase_env    

class SearchAgent(Agent):
    r"""
    Args:

        rag_config (RAGConfig): 包含 RAG 查询相关的配置
    """
    
    def __init__(
            self,

            rag_config: RAGConfig,  # 新增 RAG 配置
            model_config: Optional[Any] = None
    ) -> None:
        # 确保 rag_config 中有正确的 llm_model，否则使用默认模型
        model_name = getattr(rag_config, 'llm_model', 'gpt-4o-mini')  # 默认使用 gpt-4o-mini
        super().__init__(model=model_name, model_config=model_config)

        self.rag_config = rag_config  # 使用 RAG 配置
        self.index = None 
        self.memory_index=None # 索引将在后续通过 chat_env 中的 rag_data 构建

    def initialize_rag_index(self, rag_data):
        """
        初始化 RAG 索引，使用从 chat_env 中传递的 rag_data 构建索引
        """
        if not os.path.exists(self.rag_config.persist_dir):
            # 使用 JSONNodeParser 解析 rag_data 并构建节点
            parser = JSONNodeParser()
            
            documents = [Document(id=str(i), text=json.dumps(doc)) for i, doc in enumerate(rag_data)]
            nodes=parser.get_nodes_from_documents(documents)


            # 通过节点构建索引
            self.index = VectorStoreIndex(
                nodes,
                llm=OpenAI(
                model=self.rag_config.llm_model,  # 使用 RAG 配置中的模型
                temperature=self.rag_config.temperature  # 使用 RAG 配置中的 temperature
                )
            )

            # 持久化索引
            self.index.storage_context.persist(persist_dir=self.rag_config.persist_dir)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=self.rag_config.persist_dir)
            self.index = load_index_from_storage(storage_context)
            
    def initialize_memory_index(self, memory_data):
        """
        初始化 RAG 索引，使用从 chat_env 中传递的 rag_data 构建索引
        """
        if not os.path.exists(self.memory_config.persist_dir):
            # 使用 JSONNodeParser 解析 rag_data 并构建节点
            parser = JSONNodeParser()
            
            documents = [Document(id=str(i), text=json.dumps(doc)) for i, doc in enumerate(memory_data)]
            nodes=parser.get_nodes_from_documents(documents)


            # 通过节点构建索引
            self.memory_index = VectorStoreIndex(
                nodes,
                llm=OpenAI(
                model=self.memory_config.llm_model,  # 使用 RAG 配置中的模型
                temperature=self.memory_config.temperature  # 使用 RAG 配置中的 temperature
                )
            )

            # 持久化索引
            self.memory_index.storage_context.persist(persist_dir=self.memory_config.persist_dir)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=self.memory_config.persist_dir)
            self.memory_index = load_index_from_storage(storage_context)

    
    def rag_query_normal(self, task_query) -> list:
        """
        执行 RAG 查询，从索引中检索与查询相关的文档内容。
        """
        if self.index is None:
            raise RuntimeError("RAG Index is not initialized. Call initialize_rag_index() first.")
        
        
        if isinstance(task_query, dict): # Convert dictionary to string format 
            query = ', '.join(f'{k}: {v}' for k, v in task_query.items()) 
        else: query = task_query


        print(f"Executing RAG query: {query}")  # 调试输出


        retriever = self.index.as_retriever(
            retriever_mode="llm",
            similarity_top_k=self.rag_config.similarity_top_k  # 使用 RAG 配置中的 top_k
        )
        nodes = retriever.retrieve(query)
        results = [node.get_text() for node in nodes]


        return "\n".join(results)


    
    def memory_query(self, task_query) -> list:
        """
        执行 RAG 查询，从索引中检索与查询相关的文档内容。
        """
        if self.memory_index is None:
            raise RuntimeError("RAG Index is not initialized. Call initialize_rag_index() first.")
        
        
        if isinstance(task_query, dict):
            query = ', '.join(f'{k}: {v}' for k, v in task_query.items())
        else: query = task_query

       

        retriever = self.memory_index.as_retriever(
            retriever_mode="llm",
            similarity_top_k=self.memory_config.similarity_top_k  # 使用 Memory 配置中的 top_k

        )
        nodes = retriever.retrieve(query)
        # 使用节点的 'score' 属性计算检索结果的相似度
        
        similarity_scores = [node.score for node in nodes]
        
        threshold=self.memory_config.threshold
        
        if max(similarity_scores) < threshold:
            print("No relevant documents found with high enough similarity in memory. Switching to rag.")
            return None
        print(f"Executing RAG query: {query}")  # 调试输出
        ids=[]
        for node in nodes:
            text = node.get_text()
            parsed_data = parse_raw_text_to_dict_memory(text)
            id = parsed_data.get("id")
            if id:
                ids.append(id)
        return ids


    
    def generate_result_from_memory(self, query, memory_data: list) :

        ids=self.memory_query(query)
        if ids is None:
            return None
        # 在 detailed_data 中找到这些 ID 的详细信息
        detailed_infos = find_detailed_info_by_ids(ids, memory_data)
        # 根据 accuracy 进行排序和选择
        selected_info = select_by_accuracy(detailed_infos)
        # print(f"Selected info: {selected_info}")  # 调试输出
        return selected_info

    def generate_response(self,  messages, phase_env) -> dict:
        """
        根据内存和 JSON 知识库生成响应。每次传入一个 library，并缓存一项。如果没有缓存，则创建一个新的 library。
        """
        task_query = phase_env.get('query')  # 获取用户查询
        rag_data = phase_env.get('rag_data')  # 从 phase_env 中获取 RAG 数据
        memory_index=phase_env.get('memory_index')
        memory_data = phase_env.get('memory_data')  # 从 phase_env 中获取 Memory 数据


        # 获取全局的 all_libraries 中的 library，如果没有则为 None
        library = phase_env.get('library')

        if library:
            print(f"Using cached library: {library}")  # 调试输出
            library_str = json.dumps(library)  # 将字典转换为字符串
            return {"source": "library", "result": library_str}, phase_env


        if self.memory_index is None:
            self.initialize_memory_index(memory_index)



        # 先去memory中检索如果没有再去rag中检索
        memory_result = self.generate_result_from_memory(task_query, memory_data)
        if memory_result:
            # print(f"Using cached memory result: {memory_result}")  # 调试输出
            memory_result_str = json.dumps(memory_result)  # 将字典转换为字符串
            # print(memory_result_str)
            return {"source": "library", "result": memory_result_str}, phase_env
        


        if self.index is None:
            self.initialize_rag_index(rag_data)
        # Step 1: 如果缓存的 library 存在，直接使用它并将其作为字符串传递

        
        #没有就先生成一个rag的结果
        library = self.rag_query_normal(task_query)
        if library:
            print(f"RAG result found for query '{task_query}': {library}")  # 调试输出
            
            return {"source": "rag_knowledge", "result": library}, phase_env

        # 没有找到合适的记忆或知识库结果，返回 None
        print(f"No result found for query '{task_query}' in memory or RAG.")  # 调试输出
        return {"source": "none", "result": None}, phase_env

        


class CodeAgent(Agent):

    def __init__(
            self,
            model: Optional[str] = "gpt-4o-mini",
            model_config: Optional[Any] = None,
            
    ) -> None:
        super().__init__(model, model_config)

    def generate_response(self, messages, phase_env) -> str:
        if phase_env['model'] != "Default" and phase_env['model'] != self.model_name:
            self.model_name = phase_env['model']
            self.model_backend = ModelFactory.create(self.model_name, self.model_config.__dict__)
        
        
        
        content =  self.model_backend.run(messages=messages).choices[0].message.content.strip()
        try:
            content = content.split("```python\n")[1].split("\n```")[0]


            return content, phase_env
        except:
            # 如果没有找到 python代码块，记录调试信息，但不抛出异常
            print(f"Warning: Content does not contain a valid Python code block. Full content: {content}")
            # 返回完整的内容或继续处理，允许下一个任务继续执行
            return content, phase_env
        
class ExecuteAgent(Agent):

    def __init__(
            self,
            model: Optional[str] = "gpt-4o-mini",
            model_config: Optional[Any] = None,

    ) -> None:
        super().__init__(model=model, model_config=model_config)


    def execute_code(self, code, lock) -> str:
        with lock:
            docker_command = self.get_docker_command(code)
            try:
                # 使用 subprocess 运行 Docker 命令，并传递输入
                print(code)  # 打印 代码
                result = subprocess.run(docker_command, capture_output=True, text=True, check=True, timeout=600)
                return result.stdout
            except subprocess.CalledProcessError as e:
                print(f"Error occurred: {e}")
                print(f"STDOUT: {e.stdout}")  # 打印标准输出
                print(f"STDERR: {e.stderr}")  # 打印标准错误输出
                return f"Execution failed: {e.stderr}"  # 返回错误信息
            except subprocess.TimeoutExpired as e:
                print(f"Timeout occurred: {e}")
                return f"Execution timed out"  # 返回超时信息
        
    def get_docker_command(self, code) -> List:
        """
        根据代码内容返回相应的 Docker 命令。
        """
        encoded_code = base64.b64encode(code.encode()).decode()
        if 'autogl' in code:
            docker_command = [
                "docker", "exec", "-it", "graphteam", 
                "bash", "-c", "source activate autogl && python /root/execute_code.py --code " + encoded_code
            ]
        elif 'karateclub' in code:
            docker_command = [
                "docker", "exec", "-it", "graphteam", 
                "bash", "-c", "source activate karateclub && python /root/execute_code.py --code " + encoded_code
            ]
        else:
            docker_command = [
                "docker", "exec", "-it", "graphteam", 
                "bash", "-c", "source activate pyg && python /root/execute_code.py --code " + encoded_code
            ]
        return docker_command
    
    def generate_response(self, messages, phase_env) -> str:
        """
        生成响应，执行代码逻辑，并根据任务结果更新 MemoryManager。
        """
        if phase_env['model'] != "Default" and phase_env['model'] != self.model_name:
            self.model_name = phase_env['model']
            self.model_backend = ModelFactory.create(self.model_name, self.model_config.__dict__)

        lock = phase_env['lock']
        code = phase_env['Codes']
        phase_prompt = phase_env['phase_prompt']

        result = self.execute_code(code, lock)
        if result.startswith("Success: ") or result.startswith("/") and "Success: " in result:
            # # 更新任务状态为 "complete"，并将结果保存到 MemoryManager 中
            if result.startswith("/"):
                result = "Success: " + result.split("Success: ")[1]
            if len(messages) > 1000:
                messages = messages[:1000]
            phase_env['Messages'] = messages
            return result, phase_env

        # 处理错误，避免 KeyError
        max_retry = phase_env['max_retry']
        retry = 0
        if "Execution timed out" in result:
            error_message = result
        elif "Error:" in result:
            error_message = result.split("Error:")[1]
        else:
            error_message ="Unknown error occurred"        
        while retry < max_retry:
            # 使用 error_message，并在无错误时提供默认消息
            messages.append({
                "role": "user",
                "content": phase_prompt.format(error_message=error_message)
            })
            print(f"Retry {retry + 1}/{max_retry} for phase")
            # print(f"Retrying messages: {messages}")
            code = self.model_backend.run(messages=messages).choices[0].message.content.strip()
            try:
                code=extract_code(code)
            except:
                print(f"content does not contain a valid Python code block. full content: {code}")
            print(f"Generated code: {code}")
            phase_env['Codes'] = code
            retry += 1
            result = self.execute_code(code, lock)
            if result.startswith("Success: "):
                if len(messages) > 1000:
                    messages = messages[:1000]
                
                phase_env['Messages'] = messages
                return result, phase_env
       
            messages.extend([
                {"role": "user", "content": code},
                {"role": "user", "content": phase_prompt.format(error_message=result)}
            ])
        phase_env['Messages'] = messages

        return result, phase_env
    
class ReasoningAgent(Agent):

    def __init__(
            self,
            model: Optional[str] = "gpt-4o-mini",
            model_config: Optional[Any] = None,
    ) -> None:
        super().__init__(model, model_config)

    def generate_response(self, messages, phase_env) -> str:
        if phase_env['run'] == True:
            return None, phase_env
        if phase_env['model'] != "Default" and phase_env['model'] != self.model_name:
            self.model_name = phase_env['model']
            self.model_backend = ModelFactory.create(self.model_name, self.model_config.__dict__)
        content = self.model_backend.run(messages=messages).choices[0].message.content.strip()
        try:
            content = content.split("###output:\n")[1]
            return content, phase_env
        except:
            # 如果没有找到 output代码块，记录调试信息，但不抛出异常
            print(f"Warning: Content does not contain a valid Python code block. Full content: {content}")
            # 返回完整的内容或继续处理，允许下一个任务继续执行
            return content, phase_env
        
class CorrectionAgent(Agent):

    def __init__(
            self,
            model: Optional[str] = "gpt-4o-mini",
            model_config: Optional[Any] = None,
    ) -> None:
        super().__init__(model, model_config)

    def generate_response(self, messages, phase_env):
        if phase_env['Output_Format'] is None or phase_env['Output_Format'] == "None":
            return phase_env['Output'], phase_env
        # 检查并设置模型
        if phase_env['model'] != "Default" and phase_env['model'] != self.model_name:
            self.model_name = phase_env['model']
            self.model_backend = ModelFactory.create(self.model_name, self.model_config.__dict__)

        need_correction = True
        max_retry = phase_env['max_retry'] 
        retry = -1
        while need_correction and retry < max_retry:
            # 获取模型生成的内容
            phase_prompt = phase_env['phase_prompt'].format(output=phase_env['Output'],
                                                            output_format=phase_env['Output_Format'],
                                                            assistant_role=phase_env['assistant_role'])
            messages.append({"role": "user", "content": phase_prompt})
            print(f"Retry {retry + 1}/{max_retry} for phase")
            # print(f"Retrying messages: {messages}")
            content = self.model_backend.run(messages=messages,response_format={"type": "json_object"}).choices[0].message.content.strip()
            try:
                if content.startswith("```json\n"):
                    content = content.split("```json\n")[1].split("\n```")[0]
                # print(content)
                json_response = json.loads(content)
            except:
                # 如果没有抓取到format，记录调试信息，但不抛出异常
                print(f"Warning: Content does not contain a valid format. Full content: {content}")
                # 返回完整的内容或继续处理，允许下一个任务继续执行
                return content, phase_env  
            need_correction = json_response['need_adjustment']
            phase_env['Output'] = json_response['output']
            messages.pop()
            retry += 1
        return json_response['output'], phase_env