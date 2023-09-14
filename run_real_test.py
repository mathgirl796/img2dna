import subprocess
import time

# 创建日志文件夹
log_dir = f'log/{time.time_ns()}'
subprocess.run(f'mkdir -p {log_dir}', shell=True)

# 创建进程列表
processes = []
for idx in range(10):
    image_id = idx
    # 构造要执行的命令，例如这里是一个简单的打印语句
    command = ['python', 'coder.py', '-i', str(image_id)]
    # 指定输出文件名
    output_file = f'{log_dir}/{time.time_ns()}-{image_id}.log'
    fp = open(output_file, 'w')
    # 启动子进程
    process = subprocess.Popen(command, stdout=fp, stderr=subprocess.STDOUT)
    processes.append(process)

# 等待所有进程完成
for process in processes:
    process.wait()

# 关闭文件对象
for process in processes:
    process.stdout.close()