from pathlib import Path
from typing import Optional, Tuple, Union
import warnings
from functools import wraps


def capture_warnings(func):
    """
    捕获函数运行时的所有 warning, 并返回result+warnings
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # 捕捉所有 warning
            result = func(*args, **kwargs)
            warning_messages = [str(warning.message) for warning in w]
            return {"result": result, "warnings": warning_messages}

    return wrapper


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
