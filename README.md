# PR review

本项目主要实现使用大模型对git pr进行review, 可以指定模版输出review结果，要求支持github、gitlab、阿里的codeup

使用langchain的deepagents实现AI review相关功能, 同时需要把review的结果写回github、gitlab、codeup的PR上

## 调用方法 

### shell调用方法
```shell
python pr.py PR链接
```

### api接口

提供api接口接收 github、gitlab、codeup的回调自动触发


### Langfuse

```
langfuse = Langfuse(
  secret_key="sk-lf-8ef03ee1-5012-40f0-8500-d0386c169813",
  public_key="pk-lf-a028df2c-6bfa-45fc-9924-b79c7420912b",
  host="http://192.168.1.236:30004"
)
```