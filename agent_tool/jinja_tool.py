import os
import json
import datetime
import re
from jinja2 import Environment, FileSystemLoader, select_autoescape
import pdfkit

path_wkhtmltopdf = r'G:\wkhtmltopdf\bin\wkhtmltopdf.exe'  # 请根据你实际安装路径修改
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# 使用配置
def clean_json_string(data):
    if isinstance(data, str):
        # 去掉 Markdown 格式 ```json 和 ```，保留纯 JSON 内容
        cleaned = re.sub(r"^```(?:json)?\n?", "", data.strip())
        cleaned = re.sub(r"\n?```$", "", cleaned.strip())
        return cleaned
    return data

def pdf_output_tool(data):
    print(f"pdf_output_tool 输入类型: {type(data)}, 内容前100字符: {repr(str(data)[:100])}")

    if isinstance(data, str):
        if not data.strip():
            raise ValueError("输入字符串为空，不能解析 JSON")
        cleaned_data = clean_json_string(data)
        resume_data = json.loads(cleaned_data)
    elif isinstance(data, dict):
        resume_data = data
    else:
        raise TypeError(f"不支持的输入类型: {type(data)}")
    """
    使用 Jinja2 + WeasyPrint 渲染 HTML 简历并生成 PDF 文件。
    参数: data (str): 简历数据，JSON 字符串形式
    返回:str: 生成的 PDF 文件路径
    """
    # 解析 JSON 数据

    # 设置 Jinja2 环境
    template_dir = "agent_tool"
    template_name = "complex_resume_template.html"
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )
    template = env.get_template(template_name)

    if "resume" in resume_data:
        resume_dict = resume_data["resume"]
    elif "data" in resume_data:
        resume_dict = resume_data["data"]
    elif 'content' in resume_data:
        resume_dict = resume_data['content']
    else:
        raise ValueError("resume_data 中缺少 resume 或 data 字段")

    # 渲染 HTML
    html_content = template.render(resume=resume_dict)

    # 生成 PDF 文件路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"resume_{timestamp}.pdf")

    # 渲染 PDF
    pdfkit.from_string(html_content, pdf_path,configuration=config)
    print(pdf_path)
    return f"返回用户时要返回{pdf_path}"

