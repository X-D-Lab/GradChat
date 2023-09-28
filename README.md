<h1 align="center">GradChat(锦鲤): 考试大模型</h1>  

## 推理脚本  

```
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

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
