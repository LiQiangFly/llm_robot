from sys_prompt import *
import subprocess
import time

def record(MIC_INDEX="default", DURATION=5):

    print('开始 {} 秒录音'.format(DURATION))

    OUTPUT_FILE = 'temp/speech_record.wav'  # 输出文件

    # 构建 rec 命令
    command = ['rec', 
               '-r', '16k', 
               '-c', '1', 
               '-b', '16', 
               '-e', 'signed-integer', 
               '-t', 'wav', 
               OUTPUT_FILE
               ]

    # 启动录音子进程
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 等待指定的录音时间
    time.sleep(DURATION)

    # 发送停止信号
    proc.terminate()

    # 等待子进程结束
    proc.wait()
    print('录音结束')


def speech_recognition_cpp(audio_path='temp/speech_record.wav'):

    print('语音识别中...')
    initial_prompt = "以下是普通话的句子，这是一段语音识别。"
    model = "./whisper/ggml-small.bin"

    # 定义命令及其参数
    command = ['./whisper/main',
               '-m', model,
               '-f', audio_path,
               '--output-txt',
               '-l', 'auto',
               '--prompt', initial_prompt
               ]

    # 执行命令
    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    end_time = time.time()

    a = result.stdout

    time_stamp_end = a.find(']') + 3  # 跳过时间戳和紧跟的3个空格
    _result = a[time_stamp_end:]
    _result = _result.strip()

    print('用户指令: ', _result)
    print(f"whisper耗时: {end_time - start_time:.2f}秒\n")

    return _result


def llm_qwen(PROMPT='我想飞，你有什么办法没？'):
    # 定义命令及其参数
    command = ['./qwen/main', 
               '-m', './qwen/qwen7b-ggml.bin', 
               '--tiktoken', './qwen/qwen.tiktoken', 
               '-p', PROMPT
               ]

    # 执行命令
    print('任务规划中...')
    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    end_time = time.time()

    _result = result.stdout
    _result = _result.strip()

    print('任务编排: \n', _result)
    print(f"llm_qwen耗时: {end_time - start_time:.2f}秒")

    # print(result.stdout)
    return _result


if __name__ == '__main__':

    DURATION = 10
    record(DURATION=DURATION)  # 录音
    usr_prompt = speech_recognition_cpp()  # 语音识别
    prompt = sys_prompt + usr_prompt
    agent_plan_output = llm_qwen(prompt)  # 大语言模型进行任务规划