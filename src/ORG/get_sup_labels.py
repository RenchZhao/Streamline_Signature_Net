import os
import xml.etree.ElementTree as ET

def parse_mrml_fiber(file_path):
    fiber_vtp_ls = []
    try:
        # 解析 MRML 文件
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 打印根元素的标签
        print(f"Root tag: {root.tag}")

        # 遍历根元素的子元素
        for child in root:
            if str(child.tag)=='FiberBundleStorage':
                # 获得指向的vtp文件
                fiber_vtp_ls.append(str(child.attrib['fileName']))
        return fiber_vtp_ls
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except ET.ParseError as e:
        print(f"Error parsing the MRML file: {e}")

if __name__=='__main__':

    dir_path = '/path/to/your/dataset'
    sup_files=[]
    for file in os.listdir(dir_path):
        if 'Sup' in file and file.endswith('mrml'):
           sup_files = sup_files + parse_mrml_fiber(os.path.join(dir_path,file))
    print(sup_files)
    print(len(sup_files))