
class BaseWrapper:
    def __getattr__(self, name):
        """在访问不存在的属性时调用"""
        return getattr(self.env, name)

    def __setattr__(self, name, value):
        """在设置属性时调用"""
        if name == "env":
            # 设置包装的对象
            object.__setattr__(self, name, value)
        else:
            # 设置包装器的属性
            setattr(self.env, name, value)
