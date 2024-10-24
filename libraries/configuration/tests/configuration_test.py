from _configuration.config import Config


class TestConfig(Config):
    def __new__(cls, *args, **kwargs):
        instance = super(TestConfig, cls).__new__(cls)
        # Handle TestConfig-specific initialization before parent's __init__
        instance._testconfig_init(
            kwargs.pop("initial_data", None), kwargs.pop("arguments", None)
        )
        return instance

    def _testconfig_init(self, initial_data=None, arguments=None):
        if initial_data:
            self.load_initial_data(initial_data)
        if arguments:
            self.override_arguments(arguments)

    def __init__(self, *args, **kwargs):
        # Pass only valid arguments to the parent class
        super(TestConfig, self).__init__(*args, **kwargs)

    def load_initial_data(self, initial_data):
        for section, key_value_pairs in initial_data.items():
            for kv in key_value_pairs:
                for key, value in kv.items():
                    self.override_value(section, key, value)


def test_some_function2():
    test_args = ["", '--config-override="section=section,option=key,value=value2"']
    config = TestConfig(
        initial_data={"section": [{"key": "value"}]}, arguments=test_args
    )
    print(config.config)
    assert config.config["section"].get("key") == "value2"
