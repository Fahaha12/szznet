import os

def list_filenames_without_extension(directory, output_file):
    with open(output_file, 'w') as file:
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                name_without_extension = os.path.splitext(filename)[0]
                file.write(name_without_extension + '\n')

# 调用函数，参数为你要遍历的目录和输出文件的路径
list_filenames_without_extension('D:\深度学习\szz-net\BraTS\JPEGImages', 'filenames.txt')