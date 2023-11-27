<h1 align="center">GradChat(锦鲤): 考试大模型</h1>  

## 🔎模型介绍
人生本身就是一次毕业之旅，旅行途中程，我们每个人都要经历一场又一场的毕业小考，有制度体质内的幼托、小学、中学、大学的归档考核，也有职业升迁与行业赛道变轨的阶段考验，每一次毕业又是新的旅程的开始，出色地完成每一次毕业考核与新旅程定向终将铸就美好的人生风景。

**GradChat（锦鲤）**作为一款贯通学习与工作全生命周期的AI智能助手，不仅拥有丰富的学科知识和教育场景，还致力于为用户提供完整的**生涯规划指导**，包括**升学**、**就业**、**出国留学**、**考公考编**、**职业资格考证**等。

## 🦊模型功能
**GradChat**除了在教育场景下提供多学科**自动出题、作业批改、课程辅导、自由问答、外语陪练**等丰富功能，也为**中考、高考、考研、考博、出国留学等考生**提供**院校信息与志愿填报建议**，打破知识盲区与报考壁垒，还可为**从业人士**提供**就业指导与职业规划建议**。

🙅‍ 此外，本模型还依托[MindChat（漫谈）心理大模型](https://github.com/X-D-Lab/MindChat)强大的**情感支持**和[Sunsimiao（孙思邈）医学大模型](https://github.com/X-D-Lab/Sunsimiao)的**医疗服务**，为莘莘学子和各行各业工作者提供身心健康的**全面服务**。


👏 更为优质、安全、温暖的模型正在赶来的路上，欢迎关注：[GradChat Github](https://github.com/X-D-Lab/GradChat)



## 推理脚本  

```
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from modelscope import snapshot_download

model_dir = snapshot_download("thomas/test", revision = 'v1.0.10')

tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", bf16=True, trust_remote_code=True).eval()

model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True) 

response, history = model.chat(tokenizer, "你好", history=None)
print(response)

```

## 👨‍💻 研发团队

本项目由**华东理工大学 X-D Lab**课题组发起:
| 主要分工 | 参与同学 |
| :----: | :---- |
| 模型训练 | [颜鑫](https://github.com/thomas-yanxin)、[唐井楠](https://github.com/jingnant) |
| 模型测试 |  |
| 数据构建 |  |
