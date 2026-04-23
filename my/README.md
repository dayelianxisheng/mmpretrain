## 分析工具

### 日志分析工具
```
python tools/analysis_tools/analyze_logs.py plot_curve  \
    ${JSON_LOGS}  \
    [--keys ${KEYS}]  \
    [--title ${TITLE}]  \
    [--legend ${LEGEND}]  \
    [--backend ${BACKEND}]  \
    [--style ${STYLE}]  \
    [--out ${OUT_FILE}]  \
    [--window-size ${WINDOW_SIZE}]
```
```
所有参数的说明：

json_logs : 模型配置文件的路径（可同时传入多个，使用空格分开）。
--keys : 分析日志的关键字段，数量为 len(${JSON_LOGS}) * len(${KEYS}) 默认为 ‘loss’。
--title : 分析日志的图片名称，默认使用配置文件名， 默认为空。
--legend : 图例的名称，其数目必须与相等len(${JSON_LOGS}) * len(${KEYS})。 默认使用 "${JSON_LOG}-${KEYS}".
--backend : matplotlib 的绘图后端，默认由 matplotlib 自动选择。
--style : 绘图配色风格，默认为 whitegrid。
--out : 保存分析图片的路径，如不指定则不保存。
--window-size: 可视化窗口大小，如果没有指定，默认为 '12*7'。如果需要指定，需按照格式 'W*H'。

```
#### 损失曲线图
```python tools/analysis_tools/analyze_logs.py plot_curve your_log_json --keys loss --legend loss```



### 模型复杂度分析

#### 计算 FLOPs 和参数数量 
```python tools/analysis_tools/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]```\
例子\
```python tools/analysis_tools/get_flops.py work_dirs/mobilenetv2_voc/mobilenetv2_voc.py ```

### 混淆矩阵
```
python tools/analysis_tools/confusion_matrix.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT} \
    [--show] \
    [--show-path] \
    [--include-values] \
    [--cmap ${CMAP}] \
    [--cfg-options ${CFG-OPTIONS}]
```

```
所有参数的说明：
config：模型配置文件的路径。
checkpoint：权重路径。
--show：是否展示混淆矩阵的 matplotlib 可视化结果，默认不展示。
--show-path：如果 show 为 True，可视化结果的保存路径。
--include-values：是否在可视化结果上添加数值。
--cmap：可视化结果使用的颜色映射图，即 cmap，默认为 viridis。
--cfg-options：对配置文件的修改，参考学习配置文件。
```

### 数据集验证
在 MMPretrain 中，tools/misc/verify_dataset.py 脚本会检查数据集的所有图片，查看是否有已经损坏的图片。
``` 
python tools/print_config.py \
    ${CONFIG} \
    [--out-path ${OUT-PATH}] \
    [--phase ${PHASE}] \
    [--num-process ${NUM-PROCESS}]
    [--cfg-options ${CFG_OPTIONS}]
```

```
所有参数说明:

config : 配置文件的路径。

--out-path : 输出结果路径，默认为 ‘brokenfiles.log’。

--phase : 检查哪个阶段的数据集，可用值为 “train” 、”test” 或者 “val”， 默认为 “train”。

--num-process : 指定的进程数，默认为 1。

--cfg-options: 额外的配置选项，会被合入配置文件，参考教程 1：如何编写配置文件。

```
示例
```
python tools/misc/verify_dataset.py configs/t2t_vit/t2t-vit-t-14_8xb64_in1k.py --out-path broken_imgs.log --phase val --num-process 8
```

## 优化器参数策略可视化
检查优化器的超参数调度器（无需训练），支持学习率（learning rate）和动量（momentum）。
```
python tools/visualization/vis_scheduler.py \
    ${CONFIG_FILE} \
    [-p, --parameter ${PARAMETER_NAME}] \
    [-d, --dataset-size ${DATASET_SIZE}] \
    [-n, --ngpus ${NUM_GPUs}] \
    [-s, --save-path ${SAVE_PATH}] \
    [--title ${TITLE}] \
    [--style ${STYLE}] \
    [--window-size ${WINDOW_SIZE}] \
    [--cfg-options]

```
```
所有参数的说明：

config : 模型配置文件的路径。

-p, parameter: 可视化参数名，只能为 ["lr", "momentum"] 之一， 默认为 "lr".

-d, --dataset-size: 数据集的大小。如果指定，build_dataset 将被跳过并使用这个大小作为数据集大小，默认使用 build_dataset 所得数据集的大小。

-n, --ngpus: 使用 GPU 的数量，默认为 1。

-s, --save-path: 保存的可视化图片的路径，默认不保存。

--title: 可视化图片的标题，默认为配置文件名。

--style: 可视化图片的风格，默认为 whitegrid。

--window-size: 可视化窗口大小，如果没有指定，默认为 12*7。如果需要指定，按照格式 'W*H'。

--cfg-options: 对配置文件的修改，参考学习配置文件。

```
### 开始训练前可视化学习率曲线
``` python tools/visualization/vis_scheduler.py configs/swin_transformer/swin-base_16xb64_in1k.py --dataset-size 1281167 --ngpus 16```