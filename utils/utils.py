import configparser
import json
from pathlib import Path
from typing import Any, Optional, Tuple, Union


def prepare_directories(
    input_file_path: Union[str, Path],
    output_file_name: Optional[Union[Path, str]] = None,
    output_folder: Optional[Union[Path, str]] = None,
) -> Tuple[Path, Path]:
    """
    不传输出文件夹则会将工作目录设置在当前pdf同级且同名的文件夹下，图像会在这个文件夹的images/里
    """
    input_path = Path(input_file_path)
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Can not find file: {input_file_path}")

    # 处理输出文件夹
    if output_folder is None:
        # 默认输出文件夹：输入文件同级目录下创建同名文件夹
        output_dir = input_path.parent / input_path.stem
    else:
        # 使用指定的输出文件夹
        output_dir = Path(output_folder)

    # 处理输出文件名
    if output_file_name is None:
        # 默认输出文件名：使用输入文件的主文件名
        markdown_filename = f"{input_path.stem}.md"
    else:
        # 使用指定的输出文件名，确保有.md扩展名
        output_file_path = Path(output_file_name)
        if output_file_path.suffix.lower() != ".md":
            markdown_filename = f"{output_file_path.stem}.md"
        else:
            markdown_filename = output_file_path.name
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    # 创建输出Markdown文件路径
    output_markdown_path = output_dir / markdown_filename
    # 创建图片输出目录（在输出目录下的images子目录）
    image_output_dir = output_dir / "images"
    image_output_dir.mkdir(parents=True, exist_ok=True)

    return output_markdown_path, image_output_dir


def get_config_value(
    config: dict, section: str, key: str, default: Any, type_func=None
) -> Any:
    """从配置字典中获取值并转换类型
    Args:
        config: 配置字典
        section: 配置项所在的节
        key: 配置项的键
        default: 默认值
        type_func: 类型转换函数
    Returns:
        Any: 配置值
    """
    try:
        value = config[section][key]
        if type_func:
            if type_func == bool:
                # 处理布尔值：支持 true/false, True/False, 1/0
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            return type_func(value)
        return value
    except (KeyError, TypeError):
        return default


def load_config(config_path: Union[str, Path]) -> dict:
    """加载配置文件，支持 json / yaml / ini
    Args:
        config_path: 配置文件路径（.json | .yml/.yaml | .ini）
    Returns:
        dict: 配置字典（INI 返回 {section: {key: value}} 结构）
    """
    path = Path(config_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if suffix in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise ImportError(
                "PyYAML is required to load YAML configs. Install with: pip install pyyaml"
            ) from e
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError("YAML root must be a mapping (dict)")
            return data
    if suffix == ".ini":
        parser = configparser.ConfigParser()
        parser.read(path, encoding="utf-8")
        # 转换为 {section: {key: value}} 的普通字典
        return {
            section: {k: v for k, v in parser.items(section)}
            for section in parser.sections()
        }

    raise ValueError(f"Unsupported config format: {suffix}")
