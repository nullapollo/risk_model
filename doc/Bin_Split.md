## 评分卡分箱

### 为什么要分箱
- 避免数据无意义的波动对评分的影响
- 避免极端值对评分的影响
- 所有变量在尺度上进行统一（分箱woe编码）

### 分箱类型
- 无监督
  - 等距
  - 等频
  - 聚类
- 有监督（Heuristic Algorithm，计算量大）
  - Best-KS：自顶向下
  - $\chi^2$分箱：自底向上



### $\chi^2$ 分箱

基本思想：自底向上，变量的每个取值为独立一组，然后根据组间的排序关系，按照相邻组的 $\chi^2$ 值，选择最小$\chi^2$值的pair 进行合并，直到满足停止准则





### Best-KS 分箱

基本思想：自顶向下，对排序的变量，按照最佳 KS 值，选择最优切分点，对原数据进行二分，再对二分后的子集，继续该操作，直到满足停止准则