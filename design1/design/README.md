---
tasks:
- chat
model-type:
- chatglm
frameworks:
- pytorch
language:
- zh
- en
datasets:
  train:
  - Chz6569/FlexTrain-1241-25465f80-7c45-4912-81e4-05454b0f1d07
widgets:
- task: chat
  model_revision: v1.0
  enabled: false
  inputs:
  - type: text
    name: text
    title: 输入文字
    validator: {}
  - type: text-list
    name: history
    title: ""
    validator: {}
  examples:
  - name: "1"
    title: 示例1
    inputs:
    - name: text
      data: 你好
    - name: text
      data: []
    parameters: []
  parameters: []
  inferencespec:
    cpu: 4
    memory: 24000
    gpu: 1
    gpu_memory: 16000
  enable: true
  version: 1
---
# 该模型为使用ModelScope Trainer 微调的模型
 - 基础模型：[chatglm2-6b](/models/ZhipuAI/chatglm2-6b/summary)
 -  任务类型：generation
-  任务名称：[C](/flextrain/1241/4)

# 评估结果
| revision | rouge-1 | rouge-2 | rouge-l | bleu-4 |
| --- | --- | --- | --- | --- |  
| v1.0 | 37.74529130434782 | 15.02444347826087 | 31.13961956521739 | 12.68465434782609 |
# 示例代码
此模型需要用Lora方式进行推理
```python
from modelscope import Model, pipeline, read_config
from modelscope.metainfo import Models
from modelscope.swift import Swift
import torch
from modelscope.hub.snapshot_download import snapshot_download
import os.path as osp
from modelscope.swift.lora import LoRAConfig
from modelscope.utils.config import ConfigDict
from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = '请从ModelScope个人中心->访问令牌获取'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

lora_config = LoRAConfig(
    replace_modules=['attention.query_key_value'],
    rank=32,
    lora_alpha=32,
    lora_dropout=0.05
    )

model_dir = 'ZhipuAI/chatglm2-6b'
model_config = read_config(model_dir)
model_config['model'] = ConfigDict({'type': Models.chatglm2_6b})
model = Model.from_pretrained(model_dir, cfg_dict=model_config)
model = model.bfloat16()
Swift.prepare_model(model, lora_config)
# flex train 训练得到的模型'xxxxxxxxxx'，snapshot_download不支持model_revision,所以用revision
work_dir = snapshot_download('xxxxxxxxxx',revision='xxx')
state_dict = torch.load(osp.join(work_dir, 'pytorch_model.bin'))
model.load_state_dict(state_dict)
pipe = pipeline('chat', model, pipeline_name='chatglm2_6b-text-generation')
result_zh = pipe({
        'text':
        '简述曼德拉效应',
        'history': []
    })
print(result_zh['response'])
```