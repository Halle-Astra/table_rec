# 20241006

如果pycharm使用debug功能出现如下报错，原因是因为搜索路径下有`io.py`，导致与原生io模块起到冲突，尤其是win下当前文件夹会作为搜索路径，然后导致这个问题。

```
Fatal Python error: init_sys_streams: can't initialize sys standard streams
Python runtime state: core initialized
AttributeError: module 'io' has no attribute 'OpenWrapper'

Current thread 0x0001a704 (most recent call first):
<no Python frame>
```

参考帖子：https://segmentfault.com/q/1010000042834496/a-1020000042834498

尽管不能用debug，但是可以用run功能。