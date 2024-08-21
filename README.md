# llm_robot
此仓库是对@[TommyZihao](https://github.com/TommyZihao/vlm_arm/commits?author=TommyZihao)  [vlm_arm](https://github.com/TommyZihao/vlm_arm)项目的量化和压缩，原项目调用百度和零一万物商用大模型（需要花钱买`api_key`和`access_key`，然后把组装好的message丢给大模型api），需要花一点点money，我用开源模型，对其中使用到的语音识别和大语言模型进行替代，并经过量化和压缩，可以直接跑在本地cpu，整个项目大小4.84GB，效果不逊色商业模型。

目前实现版本，只有3个函数，录音`record`、语音识别`speech_recognition_cpp`和任务规划`llm_qwen`，并且集成到一个python文件中。暂时未实现`TTS`。

speech_recognition_cpp采用`whisper.cpp`（`ggml-small.bin`）的small版本，大小488MB，识别效果不错，`20s`时长语音识别大约`0.9s`即可搞定。

`llm_qwen`采用`qwen.cpp`，将`qwen-7B-chat 4-bit`量化后编译（`qwen7b-ggml.bin`），同时启动OpenBLAS库进一步加速（不过效果不显著呀），模型最终大小4.35GB，效果还不错，只是推理时间还不太行，大概要13sec。

另外，qwen-1.8B小是小，但效果不行啊！

为了进一步减少推理时间，有一些优化方向，列出来，有兴趣有条件的兄弟可以接着优化：

- 硬件。我的mac CPU12核18GB。CPU做矩阵运算，能力很弱，有条件当然用同等GPU啦
- 微调。有GPU，可以用lora试试，把sys_promt吸收到模型中去，就不用每次都要重新切词编码，这部分优化应该可以大幅度提高推理效率。

无需再花时间的部分：

- 算法层面基本没什么可以做的，已经使用了OpenBLAS，线性代数c++库，矩阵运算到极致了
- speech_recognition_cpp和llm_qwen推理部分都是经过编译的c++
- qwen-7B-chat模型量化到了最小的4个bit，不能再小啦

把sys_prompt暴露出来，大家可以根据自己的机械臂、AGV等操作控制场景，将被控设备的元api接口放到里面，按照模版将自然语言和元api进行组装，就可以实现通过语音控制设备啦。

# 项目配置

## llm_qwen

### 克隆项目到本地

[qwen.cpp](https://github.com/QwenLM/qwen.cpp)

```
git clone --recursive https://github.com/QwenLM/qwen.cpp && cd qwen.cpp
```

### 量化

#### hugging face

将Qwen-7B-Chat执行4-bit量化。

```
python3 qwen_cpp/convert.py -i Qwen/Qwen-7B-Chat -t q4_0 -o qwen7b-ggml.bin
```

#### modelscope

`-i Qwen/Qwen-7B-Chat`，这个参数是从hugging face下载模型，过程会比较慢。建议从`ModelScope`（阿里云仓库）下载

先修改下convert.py代码，在228行：

```python
from modelscope import snapshot_download
model_dir = snapshot_download(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
```

然后在命令行输入

```python
python3 qwen_cpp/convert.py -i qwen/Qwen-7B-chat -t q4_0 -o qwen7b-ggml.bin
```

官方支持的量化情况，目前支持7B和14B chat模型到q4_0、q4_1、q5_0等级别的量化

The original model (`-i <model_name_or_path>`) can be a HuggingFace model name or a local path to your pre-downloaded model. Currently supported models are:

- Qwen-7B: `Qwen/Qwen-7B-Chat`
- Qwen-14B: `Qwen/Qwen-14B-Chat`

You are free to try any of the below quantization types by specifying `-t <type>`:

- `q4_0`: 4-bit integer quantization with fp16 scales.
- `q4_1`: 4-bit integer quantization with fp16 scales and minimum values.
- `q5_0`: 5-bit integer quantization with fp16 scales.
- `q5_1`: 5-bit integer quantization with fp16 scales and minimum values.
- `q8_0`: 8-bit integer quantization with fp16 scales.
- `f16`: half precision floating point weights without quantization.
- `f32`: single precision floating point weights without quantization.

### 编译

不使用OpenBLAS

```
cmake -B build
cmake --build build -j --config Release
```

使用OpenBLAS。[需要安装这个库](https://www.openblas.net/)。

```
cmake -B build -DGGML_OPENBLAS=ON && cmake --build build -j
```

### 移植

需要移植三个文件

- `main`。将build/bin文件夹中的`main`文件拷贝到[项目](https://github.com/LiQiangFly/llm_robot)qwen`文件夹中。
- `qwen.tiktoken`。用项目中文件即可。源文件在[Hugging Face](https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen.tiktoken) or [modelscope](https://modelscope.cn/models/qwen/Qwen-7B-Chat/files)。
- `qwen7b-ggml.bin`。将编译生成的模型文件拷贝到qwen文件夹中。

## speech_recognition_cpp

[核心是whisper.cpp](https://github.com/ggerganov/whisper.cpp)。

#### 模型下载

直接在[hugging face](https://huggingface.co/ggerganov/whisper.cpp/tree/main)下载你需要的模型。将模型文件放到whisper.cpp项目的models文件夹中。我下载的是 `ggml-small.bin`。

#### 编译

```
# build the main example
make

# transcribe an audio file
./main -f samples/jfk.wav
```

命令行可选参数

```
usage: ./main [options] file0.wav file1.wav ...

options:
  -h,        --help              [default] show this help message and exit
  -t N,      --threads N         [4      ] number of threads to use during computation
  -p N,      --processors N      [1      ] number of processors to use during computation
  -ot N,     --offset-t N        [0      ] time offset in milliseconds
  -on N,     --offset-n N        [0      ] segment index offset
  -d  N,     --duration N        [0      ] duration of audio to process in milliseconds
  -mc N,     --max-context N     [-1     ] maximum number of text context tokens to store
  -ml N,     --max-len N         [0      ] maximum segment length in characters
  -sow,      --split-on-word     [false  ] split on word rather than on token
  -bo N,     --best-of N         [5      ] number of best candidates to keep
  -bs N,     --beam-size N       [5      ] beam size for beam search
  -wt N,     --word-thold N      [0.01   ] word timestamp probability threshold
  -et N,     --entropy-thold N   [2.40   ] entropy threshold for decoder fail
  -lpt N,    --logprob-thold N   [-1.00  ] log probability threshold for decoder fail
  -debug,    --debug-mode        [false  ] enable debug mode (eg. dump log_mel)
  -tr,       --translate         [false  ] translate from source language to english
  -di,       --diarize           [false  ] stereo audio diarization
  -tdrz,     --tinydiarize       [false  ] enable tinydiarize (requires a tdrz model)
  -nf,       --no-fallback       [false  ] do not use temperature fallback while decoding
  -otxt,     --output-txt        [false  ] output result in a text file
  -ovtt,     --output-vtt        [false  ] output result in a vtt file
  -osrt,     --output-srt        [false  ] output result in a srt file
  -olrc,     --output-lrc        [false  ] output result in a lrc file
  -owts,     --output-words      [false  ] output script for generating karaoke video
  -fp,       --font-path         [/System/Library/Fonts/Supplemental/Courier New Bold.ttf] path to a monospace font for karaoke video
  -ocsv,     --output-csv        [false  ] output result in a CSV file
  -oj,       --output-json       [false  ] output result in a JSON file
  -ojf,      --output-json-full  [false  ] include more information in the JSON file
  -of FNAME, --output-file FNAME [       ] output file path (without file extension)
  -ps,       --print-special     [false  ] print special tokens
  -pc,       --print-colors      [false  ] print colors
  -pp,       --print-progress    [false  ] print progress
  -nt,       --no-timestamps     [false  ] do not print timestamps
  -l LANG,   --language LANG     [en     ] spoken language ('auto' for auto-detect)
  -dl,       --detect-language   [false  ] exit after automatically detecting language
             --prompt PROMPT     [       ] initial prompt
  -m FNAME,  --model FNAME       [models/ggml-base.en.bin] model path
  -f FNAME,  --file FNAME        [       ] input WAV file path
  -oved D,   --ov-e-device DNAME [CPU    ] the OpenVINO device used for encode inference
  -ls,       --log-score         [false  ] log best decoder scores of tokens
  -ng,       --no-gpu            [false  ] disable GPU
```

#### 移植

- `ggml-small.bin`。放到项目 `whisper`文件夹中。
- `main`。编译好的 `main`文件放到 `whisper`文件夹中。

## record

录音可以直接调用操作系统函数，也可以使用pyaudio这样的库，我采用前者。

根据你自己的操作系统编写record函数，Linux、Windows和macos不同，下面是macos的录音函数代码。

```
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
```

