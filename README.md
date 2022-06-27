<!--
 * @Author: error: git config user.name && git config user.email & please set dead value or install git
 * @Date: 2022-06-27 20:54:43
 * @LastEditors: error: git config user.name && git config user.email & please set dead value or install git
 * @LastEditTime: 2022-06-27 21:29:52
 * @FilePath: \baseline_mode\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->


# Usage

 To use the code, enter the models directory and execute run_Model.py
such as:
``` bash
cd models/Caser
python run_Caser.py
```

SASRec: ```python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda```

SSE-PT: ```python3 main.py --maxlen=200 --dropout_rate 0.2 --threshold_user 1 --threshold_item 1 --device=cuda```

Note: Due to the different sample construction methods and experimental methods of different algorithms, we generate independent codes for each algorithm.

    
# Requirements
* Tensorflow 1.1+
* Python 3.6+, 
* numpy
* pandas

# ToDo List
* More models
* Code refactoring
* Support tf.data.datasets and tf.estimator


