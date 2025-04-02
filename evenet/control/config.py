# config.py

import yaml
from copy import deepcopy
from pathlib import Path
from rich import get_console
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.console import Console

from evenet.control.default_config import _DEFAULTS
from evenet.control.event_info import EventInfo


class DotDict(dict):
    """Recursive dict with attribute-style access."""

    def __init__(self, d=None):
        super().__init__()
        d = d or {}
        for k, v in d.items():
            self[k] = self._wrap(v)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"'DotDict' has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"'DotDict' has no attribute '{key}'")

    def _wrap(self, value):
        if isinstance(value, dict):
            return DotDict(value)
        elif isinstance(value, list):
            return [self._wrap(v) for v in value]
        return value

    def to_dict(self):
        result = {}
        for k, v in self.items():
            if isinstance(v, DotDict):
                result[k] = v.to_dict()
            elif isinstance(v, list):
                result[k] = [vv.to_dict() if isinstance(vv, DotDict) else vv for vv in v]
            else:
                result[k] = v
        return result

    def merge(self, override: dict):
        for k, v in override.items():
            if k in self and isinstance(self[k], DotDict) and isinstance(v, dict):
                self[k].merge(v)  # 🧠 recursive merge!
            else:
                self[k] = self._wrap(v)


class Config:
    """Singleton-style config manager with rich display."""

    def __init__(self, defaults: dict):
        self._defaults = DotDict(deepcopy(defaults))
        self._config = DotDict(deepcopy(defaults))

    def load_yaml(self, path: str | Path):
        path = Path(path)
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        # Loop over top-level blocks
        for section, content in data.items():
            if isinstance(content, dict) and "include" in content:
                include_path = path.parent / content.pop("include")
                with open(include_path, "r") as inc:
                    inc_data = yaml.safe_load(inc) or {}

                # Merge included YAML first
                merged = {**inc_data, **content}  # overwrite with inline
                self._config[section] = self._config.get(section, DotDict())
                self._config[section].merge(merged)

            else:
                # No include — just use this block directly
                self._config[section] = self._config.get(section, DotDict())
                self._config[section].merge(content)

        required = ["event_info", "resonance"]
        missing = [key for key in required if key not in self._config]
        if missing:
            raise ValueError(f"Missing required config section(s): {', '.join(missing)}")

        self._config["event_info"] = EventInfo.construct(
            config=self._config.pop("event_info"),
            resonance_info=self._config["resonance"],
        )

    def update(self, data: dict):
        self._config.merge(data)

    def to_dict(self):
        return self._config.to_dict()

    def save(self, path: str | Path):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f)

    def __getattr__(self, key):
        return getattr(self._config, key)

    def __getitem__(self, key):
        return self._config[key]

    def __str__(self):
        import pprint
        return pprint.pformat(self.to_dict(), indent=2)

    def _flatten_dict(self, d, parent_key=""):
        """Recursively flatten a nested DotDict into dot.key format"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def display(self):
        console = get_console()
        table = Table(title="Configuration", header_style="bold magenta")
        table.add_column("Parameter", justify="left", style="cyan")
        table.add_column("Value", justify="left")

        flat_current = self._flatten_dict(self._config['options'])
        flat_default = self._flatten_dict(self._defaults)

        for key in sorted(flat_current.keys()):
            val = flat_current[key]
            default_val = flat_default.get(key, None)
            style = "bold red" if val != default_val else "green"
            table.add_row(key, str(val), style=style)

        console.print(table)

        console = Console()
        tree = self.dict_to_rich_tree(self._config['resonance'])
        console.print(tree)

    def dict_to_rich_tree(self, data, tree=None):
        if tree is None:
            tree = Tree("Resonance Particles", guide_style="bold cyan")

        for key, value in data.items():
            if isinstance(value, dict):
                branch = tree.add(f"[bold yellow]{key}[/]")
                self.dict_to_rich_tree(value, branch)
            else:
                leaf = f"[green]{key}[/]: {value}"
                tree.add(leaf)
        return tree


# --- Global instance --- #
config = Config(_DEFAULTS)

if __name__ == '__main__':
    # Example usage
    # config.load_yaml("default.yaml")
    config.load_yaml("preprocess_pretrain.yaml")
    config.display()

    a=0

    # config.options.Network.hidden_dim
    # config.event_info.INPUTS.SEQUENTIAL.Source.mass
    # config.resonance.HadronicTop.'t/bqq'.Mass
