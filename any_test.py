data = {
    'a':1,
    'mw_dasjfiao':155
}
locals().update(data)
print(mw_dasjfiao)
class TestDict():
    def __init__(self) -> None:
        super().__init__()

    @property
    def configure_pulse(self):
        return {
            'T1': {
                'test_1': 1,
                'test_2': 2
            },
            'Rabi': {},
            'Ramsey': {},
            'Hahn Echo': {}
        } 

    def set_pulse_and_count(self):
        current_tab_name = 'T1'
        print(current_tab_name)
        
        _pulse_configuration = self.configure_pulse[current_tab_name]
        print(_pulse_configuration)
        
        # 将属性直接添加到当前方法的局部命名空间中
        for key, value in _pulse_configuration.items():
            setattr(self, key, value)
        
        # 访问局部命名空间中的属性值
        print(self.test_1)  # 输出设置的属性值，不使用 self

test = TestDict()
test.set_pulse_and_count()