from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any
from agent_tool.embedding import TransformerEmbedding
import pandas as pd

# 初始化 LLM
llm = ChatOpenAI(
    openai_api_key='sk-6321ebc75a0741e68625b92c48cb6e61',
    openai_api_base='https://api.deepseek.com',
    model_name='deepseek-chat',
    streaming=False,
    temperature=0.7
)

# 向量数据库设置
embedding = TransformerEmbedding()
vectorstore = Chroma(
    embedding_function=embedding,
    persist_directory='./db/chroma',
    collection_name="default"
)

# 初始化记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 提示信息配置 (保持不变)
prompt_infos = [
    {
        'name': '信息提供者',
        'description': '提供用户的个人信息',
        'prompt_template': """若知识库里有，提供以下个人信息：
             基于知识库信息{context}
            历史消息对话{chat_history}
            1. 👤 姓名（全名）：
            2. 🎓 教育背景（学校、专业、学历）：
            3. 🧑‍💼 工作经历（公司、职位、时间段）：
            4. 📚 技能与擅长领域（编程语言/工具/软技能等）：
            5. 🏆 项目经历（项目名称、职责、成果）：
            6. 🎯 职业目标或兴趣方向：
            7. 📍 当前所在地（城市/国家）：
            8. ✉️ 邮箱：
            9. ☎️ 电话号码：
            用户输入\n{query}
"""
    },
    {
        'name': '建议提供者',
        'description': '总结客户提供的问题并且给出建议',
        'prompt_template': """你是一名经验丰富的简历建议顾问，请针对以下问题提供具体、可操作的建议。要求： 基于知识库信息{context}
                            历史消息对话{chat_history}
                            1. **明确背景**：先简要总结问题的核心（如实习经历/项目经验/比赛经验/科研经历/GPA绩点等）；  
                            2. **分点建议**：给出3-5条步骤清晰的解决方案，避免空泛；  
                            3. **附加提示**：可补充常见误区或额外资源（如书籍/工具）
                            4.要给出做建议的依据
                            现在请处理用户的最新请求：\n。
                               用户输入\n{query}"""
    },
    {
        'name': '个人问答助手',
        'description': "回答客户的问题，包括查询简历内容",
        'prompt_template': """你是一个严谨的本地简历知识库问答助手，请严格遵循以下规则回答用户问题：
                                知识库信息{context}
                                历史消息对话{chat_history}
                            1. **回答依据**：  
                               - 仅基于提供的本地知识库内容作答，拒绝任何外部推测。基于知识库信息
                               - 若知识库中无明确答案，必须声明『根据现有资料无法确认』，而非猜测。   
                            2. **回答结构**：  
                               ```  
                               [总结]：用1句话直接回答核心问题（是/否/关键结论）。  
                               [依据]：引用知识库中的具体条目（如文件名称/章节/关键数据），分点列出支持证据。  
                               [扩展]：可选补充相关知识点（需明确标注来源）。  
                               ```  
                            3. **不确定性处理**：  
                               - 若用户问题存在歧义，需主动请求澄清（例：『您指的是A功能还是B功能？』）。  
                               - 对超出知识库范围的问题，应回复：『该问题暂未收录，建议通过XX渠道进一步咨询』
                               现在请处理用户的最新请求：\n。  
                                用户输入\n{query}"""
    },
    {
        'name': '简历整合者',
        'description': "如有明确需要生成简历时，基于知识库内容将信息整合为简历",
        'prompt_template': """
                            "请基于知识库信息帮我整合一份专业的中文简历内容，要求逻辑清晰、重点突出，并量化个人成就。按步骤回答：  
                                知识库信息{context}
                                历史消息对话{chat_history}
                            1. **基础信息整合**  
                               - 姓名、联系方式、求职意向等（若需补充请提示我）  

                            2. **教育背景**  
                               - 按时间倒序列出学历（学校、专业、时间段）  
                               - 补充：成绩排名/奖学金/核心课程/学术荣誉（若有）  

                            3. **工作经历**(如果有的话) 
                               - 按时间倒序排列（公司、职位、时间段）  
                               - 每段工作提炼3-5条职责，用**STAR法则**（情境-任务-行动-结果）描述  
                               - 强调可量化的成果（如提升效率X%、完成Y项目等）  

                            4. **项目经验**(如果有的话)
                               - 项目名称、角色、时间段  
                               - 说明项目目标、你的具体贡献、成果数据（如用户增长X%）

                            5. **比赛经验**
                               - 包含：比赛名称、奖项级别、参赛时间。
                               - 简述你的角色、解决的问题和最终结果（例如“第八届校赛一等奖，团队排名第1/300”）。  

                            6. **技能与证书**(如果有的话) 
                               - 分点列出硬技能（如编程语言、工具）、软技能（如团队协作）  
                               - 补充相关证书（名称+等级/有效期）  

                            7. **个人亮点优化**  
                               - 根据上述内容，提炼2-3条核心竞争力（如‘3年XX领域经验，主导过千万级项目’）  
                               - 是否需要补充行业关键词以适配ATS系统？  

                            **附加要求**：  
                            - 语言简洁，避免冗长  
                            - 重要数据加粗（如‘**营收增长30%**’）  
                            - 最终按简历常用版式分段输出" 
                            最终输出正确的JSON格式内容，并可以直接被json.loads()函数解析，
                            应该是"resume":'name','email'这样的json格式，以resume作为开头，知识库里没有的信息用空格代替
                            要严格按照这个格式，应该包含
                            resume.name                       # 姓名  
                            resume.email                      # 邮箱  
                            resume.phone                      # 电话  
                            
                            resume.education[...].school      # 学校名称  
                            resume.education[...].degree      # 学位（如 本科、硕士）  
                            resume.education[...].year        # 时间范围（如 09/2023 - 06/2027）  
                            resume.education[...].major       # 专业  
                            resume.education[...].additional_info  # 其他补充信息（如实验室经历）  
                            
                            resume.projects[...].name         # 项目名称  
                            resume.projects[...].time         # 项目时间  
                            resume.projects[...].description  # 项目描述（多行文本）  
                            resume.projects[...].technologies[...] # 技术栈列表  
                            
                            resume.experience[...].company    # 公司/组织名称  
                            resume.experience[...].position   # 担任职务  
                            resume.experience[...].period     # 时间段  
                            resume.experience[...].detail     # 经历细节描述  
                            
                            resume.skills[...].name           # 技能名称（如 Python）  
                            resume.skills[...].level          # 技能熟练度（如 熟练）  
                            
                            resume.certificates[...].title    # 证书名称  
                            resume.certificates[...].year     # 获得年份  
                            resume.certificates[...].issuer   # 发证机构（可选）  
                            resume.certificates[...].verification_link  # 认证链接（可选）  
                            
                            resume.competitions[...].name     # 比赛名称  
                            resume.competitions[...].award    # 奖项（如 一等奖）  
                            resume.competitions[...].time     # 时间  
                            resume.competitions[...].role     # 担任角色（如 负责人）  
                            resume.competitions[...].description # 项目描述  
                            
                            resume.additional_info.english_level  # 英语等级（如 CET-6）  
                            resume.additional_info.interests      # 兴趣方向（如 大语言模型、AI等）  
                            

                            现在请处理用户的最新请求：\n。  
                                用户输入\n{query}
                                --- """
    },
    {
        'name': '职业规划者',
        'description': '为客户提供个性化的关于考研、留学或就业的服务',
        'prompt_template': """你是一位资深的大学生职业与升学规划顾问，熟悉国内外考研趋势与就业市场。请根据用户知识库提供的信息，提供个性化、结构化的升学与就业双路径建议，包括但不限于以下内容：
                                知识库信息{context}
                                历史消息对话{chat_history}

                            ---

                            🎯 1. 【背景分析】  
                            - 概括用户的专业、年级、成绩、实习/项目经历、兴趣方向、能力基础等核心信息。

                            🎯 2. 【目标梳理与澄清】  
                            - 判断用户当前倾向考研还是就业，是否明确目标方向。  
                            - 如有模糊或矛盾之处，请提出澄清建议。

                            🎯 3. 【双路径规划】  
                            > 若用户目标尚不确定，请并行给出“考研路线”和“就业路线”的对比与发展建议；  
                            > 若用户目标明确，请聚焦在其目标上提供详细路径。

                            ✅ 考研路径（如适用）：  
                            - 适合的目标专业与院校建议（含选校梯度）  
                            - 初试/复试关键点与备考时间表  
                            - 如何提升背景竞争力（科研、竞赛、实习等）

                            ✅ 就业路径（如适用）：  
                            - 对应专业的热门岗位与行业分析  
                            - 简历优化与面试准备建议  
                            - 如何积累可投递项目与实习经验

                            🎯 4. 【资源推荐】  
                            - 推荐相关备考网站/书籍/课程/实习平台等，辅助用户更好规划。

                            🎯 5. 【风险与建议】  
                            - 提醒潜在风险、误区，给出调整建议；  
                            - 可鼓励用户做 SWOT 分析（优势、劣势、机会、威胁）。

                            ---

                            请注意：
                            - 回答要 **结构清晰、实用具体、逻辑严谨**；
                            - 语言需具备 **鼓励性和专业性**，帮助用户更好地看清方向；
                            - 若信息不足，可在回答前主动引导用户补充具体背景信息。


                            \n现在请处理用户的最新请求：\n。
                            用户输入\n{query}"""
    },
]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_chat_history(chat_history):
    return "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

"""调试检索与记忆模块"""
#query1='我适合去哪个国家，只需给名字'
#query2=('我上一句话说了什么')
#result=full_chain.invoke(query1)# 保存人类输入
#memory.chat_memory.add_message(HumanMessage(content=query1))

#memory.chat_memory.add_message(AIMessage(content=result.content))
#print(memory.load_memory_variables({})['chat_history'])
#print(full_chain.invoke(query2).content)

#整合为tool function
class UserFilterRetriever:
    def __init__(self, vectorstore, user_id, k=3):
        self.vectorstore = vectorstore
        self.user_id = user_id
        self.k = k

    def get_relevant_documents(self, query):
        return self.vectorstore.similarity_search(
            query,
            k=self.k,
            filter={"user_id": self.user_id}
        )
def resume_qa_tool_func(query: str) -> str:
    USER_DB_PATH = "user_db.csv"
    user_id = None
    df = pd.read_csv(USER_DB_PATH)

    for i in range(len(df['login_state'])):
        if df.loc[i, 'login_state'] == True:
            user_id = df.loc[i, 'username']
            print(user_id)
            break
    retriever = UserFilterRetriever(vectorstore, user_id)
    retriever_runnable = RunnableLambda(lambda x: retriever.get_relevant_documents(x))
    destination_chains: Dict[str, Any] = {}
    for info in prompt_infos:
        prompt = ChatPromptTemplate.from_template(info['prompt_template'])

        # 自定义处理链
        chain = (
                RunnableParallel({
                    "context": retriever_runnable | format_docs,
                    "chat_history": lambda x: get_chat_history(memory.load_memory_variables({})["chat_history"]),
                    "query": RunnablePassthrough()
                })
                | prompt
                | llm
        )
        destination_chains[info['name']] = chain

    # 默认链
    default_prompt = ChatPromptTemplate.from_template(
        "你是用户的知识库助手，虽然你目前不支持此功能，但是要根据历史记录: {chat_history} 回复他  用户输入： {query}"
    )
    default_chain = default_prompt | llm

    # 路由链
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations="\n".join(destinations))
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["query"],
        output_parser=RouterOutputParser(),
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)
    # 创建检索器


    def route_info(info: Dict) -> Any:

        if isinstance(info["destination"], dict):
            destination = info["destination"].get("destination")
        else:
            destination = info["destination"]

        if destination in destination_chains:

            return destination_chains[destination].invoke({
                "query": info["query"],
                "chat_history": info['chat_history']  # 显式注入
            })
        else:
            return default_chain.invoke({"query": info["query"], 'chat_history': info['chat_history']})

    # 完整链
    full_chain = (
            RunnableParallel({
                "query": RunnablePassthrough(),
                "destination": router_chain,
                "chat_history": lambda x: get_chat_history(memory.load_memory_variables({})["chat_history"])
            })
            | RunnableLambda(route_info)
    )

    # 执行链
    result = full_chain.invoke(query)


    # 更新记忆
    memory.chat_memory.add_message(HumanMessage(content=query))
    memory.chat_memory.add_message(AIMessage(content=result.content))

    return result.content
