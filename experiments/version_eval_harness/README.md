# 版本敏感性可执行评测:测试流程

本目录存放用于对比 **你的系统(Memix)** 与 **baseline** 的测试题与流程。
共 212 题,两个库:

| 文件 | 库 | 目标版本 | 题数 |
|---|---|---|---|
| `matplotlib_3_8_3_tasks.jsonl` | matplotlib | 3.8.3 | 167 |
| `torch_2_2_0_tasks.jsonl` | torch | 2.2.0 | 45 |

每道题的判定方式(已在出题时全部验证过):**旧 API 写法在目标版本环境中必然报错,新 API 写法运行并通过语义断言**。所以通过率直接反映系统提供"版本正确"上下文的能力。

> 以下所有命令都在仓库根目录(`LibEvolutionEval/`)下执行。本文件所在目录为
> `experiments/version_eval_harness/`。

## 1. 准备环境(一次性)

两个 pinned 环境,评分时会断言库版本,版本不对分数无效:

```bash
python3 -m venv .venv-mpl383
.venv-mpl383/bin/python -m pip install \
  -r executable_eval/requirements-matplotlib-3.8.3.txt

python3 -m venv .venv-torch220
.venv-torch220/bin/python -m pip install \
  -r executable_eval/requirements-torch-2.2.0.txt
```

## 2. 题目格式

每行一个 JSON 对象:

```json
{
  "id": "torch_symeig_removed",
  "library": "torch",
  "target_version": "2.2.0",
  "api": "torch.symeig",
  "description": "Compute the eigenvalues and eigenvectors of ...",
  "prompt": "import torch\nassert torch.__version__ == \"2.2.0\"\n\ndef f():\n    ...\n    eigen = ",
  "right_context": "\n    return eigen\n"
}
```

模型要写的代码插入在 `prompt` 末尾与 `right_context` 开头之间(即 `prompt + completion + right_context` 必须拼成合法函数)。注意 `prompt` 常常在半个语句处截断(如上例 `eigen = `),completion 应是把该行补全的表达式/语句,一般一行即可。

## 3. 用被测系统生成答案

对每道题,把 `prompt` + 插入点 + `right_context` 交给被测系统补全。
**两个系统用完全相同的题目和指令**,唯一区别是系统各自提供的上下文/记忆:

- **baseline**:正常上下文(或它自己会检索到的、可能过时的文档);
- **Memix**:解析到目标版本(matplotlib 3.8.3 / torch 2.2.0)的文档上下文。

建议的用户指令(我们跑 deepseek-v4-flash 消融时用的措辞,可直接复用):

> Complete the following Python code by replacing `<INSERT>`:
> ` ```python\n{prompt}<INSERT>{right_context}\n``` `
> Reply with only the replacement code for `<INSERT>`.

把每个系统的输出写成 predictions JSONL,每行:

```json
{"id": "torch_symeig_removed", "completion": "torch.linalg.eigh(matrix)"}
```

要求:id 必须来自题目文件;不要重复 id;缺失的 id 计为失败;
空 completion 计为失败。建议原样保存模型返回,便于事后排查。

## 4. 评分

每个库用匹配的解释器各跑一次:

```bash
# matplotlib 167 题
python3 -m executable_eval.score_predictions \
  --cases executable_eval/cases --library matplotlib \
  --predictions path/to/memix_predictions_matplotlib.jsonl \
  --python .venv-mpl383/bin/python \
  --json-report results/memix-matplotlib-score.json

# torch 45 题
python3 -m executable_eval.score_predictions \
  --cases executable_eval/cases --library torch \
  --predictions path/to/memix_predictions_torch.jsonl \
  --python .venv-torch220/bin/python \
  --json-report results/memix-torch-score.json
```

输出 `Score: x/N` 和按变更类型(change_type)的细分通过率;baseline 同理,
换 predictions 路径即可。对比两个系统的 `Score` 即得结论。

## 5. 自检(可选,强烈建议先跑)

仓库自带参考预测,用来验证环境和评分链路正常:

```bash
# 应得 167/167 与 45/45(新 API 参考全部通过)
python3 -m executable_eval.score_predictions \
  --cases executable_eval/cases --library matplotlib \
  --predictions executable_eval/examples/reference_new_predictions_matplotlib.jsonl \
  --python .venv-mpl383/bin/python
python3 -m executable_eval.score_predictions \
  --cases executable_eval/cases --library torch \
  --predictions executable_eval/examples/reference_new_predictions_torch.jsonl \
  --python .venv-torch220/bin/python

# 应得 0/167 与 0/45(旧 API 参考全部失败)
python3 -m executable_eval.score_predictions \
  --cases executable_eval/cases --library matplotlib \
  --predictions executable_eval/examples/reference_old_predictions_matplotlib.jsonl \
  --python .venv-mpl383/bin/python
python3 -m executable_eval.score_predictions \
  --cases executable_eval/cases --library torch \
  --predictions executable_eval/examples/reference_old_predictions_torch.jsonl \
  --python .venv-torch220/bin/python
```

## 注意事项

- **不要把 `executable_eval/cases/` 下的 JSON 给模型**:里面有参考答案和隐藏断言。模型只能看本目录的两个 tasks 文件。
- completion 会被当作 Python 代码直接执行,只运行你信任的输出;超时(默认 20 秒/题)不是安全沙箱。
- 模型输出若带 markdown 代码围栏,先剥离再写入 predictions。
- 消融实验(正确/错误/无/混合文档四配置,deepseek-v4-flash)的完整结果见
  `../doc_version_ablation/REPORT.md`,可作为实验设计参照。
