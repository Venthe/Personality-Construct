from python_config.config import SingleConfig, extract_ini_parameter


# def test_some_function():
#     config = SingleConfig(initial_data={"section": [{"key": "value"}]})
#     assert config.config["section"].get("key") == "value"


def test_some_function2():
    test_args = ["", '--config-override="section=section,option=key,value=value2"']
    config = SingleConfig(initial_data={"section": [{"key": "value"}]}, arguments=test_args)
    assert config.config["section"].get("key") == "value2"

# def test_extract_ini():
#     section, option, value = extract_ini_parameter("section=section,option=key,value=value2")
#     assert section=="section"
#     assert option=="key"
#     assert value=="value2"