from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent,Tool
from langchain_core.prompts import ChatPromptTemplate
from qachain import resume_qa_tool_func
from pydantic import BaseModel
from agent_tool.jinja_tool import pdf_output_tool
from langchain_community.utilities import SerpAPIWrapper

class ResumeQAInput(BaseModel):
    query: str

search = SerpAPIWrapper(serpapi_api_key='8425b0c3cf3c65de9d6419a19a686c4abf3066aa217004d03c3e0171443d4f19')
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="适用于回答当前信息、时事、学校招生等问题。输入应为完整问题或关键词。"
    ),
    Tool(
        name='resume_qa',
        func=resume_qa_tool_func,
        description="一个帮你查找信息的工具",
        args_schema=ResumeQAInput
    ),
    Tool(
        name='pdf_generater',
        func=pdf_output_tool,
        description='帮你把简历按模板输出pdf的工具',
        args_schema=ResumeQAInput
)

]



deepseek_KEY='sk-6321ebc75a0741e68625b92c48cb6e61'
deepseek_URL='https://api.deepseek.com'
llm=ChatOpenAI(
    openai_api_key=deepseek_KEY,
    openai_api_base=deepseek_URL,
    model_name='deepseek-chat',
    streaming=False,
    temperature=0.3
)
#调试方法
def func(query:str):
    print(f'成功调用{query}')
    return '信息如下'
#调试工具
llm_tool=Tool(
    name='resume_qa',
    func=func,
    description="一个帮你查找信息的工具",
    args_schema=ResumeQAInput
)




tool_names = [tool.name for tool in tools]
tool_descriptions = "\n".join([f"{tool.name}：{tool.description}" for tool in tools])

prompt = ChatPromptTemplate.from_template(
    """你是一个专业的简历处理AI助手，按以下规则操作：
        
        ### 可用工具：
        {tools}
        
        ### 必须遵守的格式（只能选择其一）：
        
        - 如需调用工具（Tool）：
        Thought: 你的思考过程
        Action: 工具名（必须是 [{tool_names}] 之一）
        Action Input: "工具输入内容"
        
        - 如已获得最终答案：
        Thought: 你的思考过程
        Final Answer: 最终回答内容
        
        ### 重要规则：
        1. 当需要用户提供信息时，调用 resume_qa 工具
        2. 如果工具返回“无法确认”或“信息不完整”，必须再次调用 resume_qa 向用户澄清
        3. 如果用户回复“本次问答结束”或“无需补充”，请用 Final Answer 返回最终总结
        4. 当用户需要查询网页时，调用 serpapi 进行查询
        5. 不要返回网址给用户自行查询，应直接整合完内容信息发给用户
        6. 回答用户的问题时要给出依据，并且要尽可能的具体
        7. 生成简历的时候，必须严格按照resume_qa返回的json格式,不能自己更改
        8. 当需要生成简历时，应该完整保留resume_qa返回来的东西，不能自己总结更改
            必须严格按照resume_qa返回的内容,不能自己编造
        9. 用户可以向你索要个人信息
        10. 当用户向你询问其他信息时才调用默认链，例如问你今天天气如何，否则都不调用默认链
        
        ### 当前任务：
        问题：{input}
        {agent_scratchpad}
        """

).partial(
    tool_names=", ".join(tool_names),
    tools=tool_descriptions
)

# ✅ 创建 Agent 和执行器
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,max_iteration=3,handle_parsing_errors=True,
                               return_intermediate_steps=True
                               )
