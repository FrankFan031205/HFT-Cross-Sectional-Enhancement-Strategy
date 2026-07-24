
import subprocess

def delete_folder_by_cmd(folder_path):
    try:
        subprocess.run(["rm", "-rf", folder_path], check=True)
        print(f"Old data {folder_path} has been cleared")
    except subprocess.CalledProcessError as e:
        print(f"删除失败，错误信息：{e}")

