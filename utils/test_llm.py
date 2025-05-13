import pytest 
from llm import ClientFactory

# @pytest.fixture
def models():
    #config_files = ["../configs/ant_config.yaml", "../configs/gemini_config.yaml", "../configs/azure_config.yaml"]
    config_files = ["../configs/azure_config.yaml"]
    return [ClientFactory.from_config(config_file) for config_file in config_files]

# TESTS FOR MODELS
def test_model_init(models):
    for model in models:
        assert model is not None

def test_model_generate_text(models):
    for model in models:
        response = model.generate_text("Hello, how are you?")
        print(response)
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

if __name__ == "__main__":
    test_model_generate_text(models())