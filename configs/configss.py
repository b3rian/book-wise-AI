import yaml

class Config:
    def __init__(self, config_path='config.yml'):
        with open(config_path, 'r') as file:
            self.cfg = yaml.safe_load(file)

        # Flatten namespaces for easier access
        self.model = self.cfg.get("model", {})
        self.training = self.cfg.get("training", {})
        self.tokens = self.cfg.get("tokens", {})
        self.paths = self.cfg.get("paths", {})

# Initialize once and reuse
config = Config()
