# Effect 指标说明 (Cramer's V / TV / JS)

本文说明 `summaries/sentiment_significance.csv` 中的 effect 指标含义、计算公式与解读方式。
所有公式均使用标准 Markdown + LaTeX 书写，包含下标、上标与矩阵。

---

## 1) 配对表与记号

对同一批样本，base 与性格模型形成**配对预测表**：

$$
\mathbf{C} =
\begin{bmatrix}
c_{11} & \cdots & c_{1K} \\
\vdots & \ddots & \vdots \\
c_{K1} & \cdots & c_{KK}
\end{bmatrix}
$$

含义：

- 行: base 模型预测标签
- 列: 性格模型预测标签
- $c_{ij}$: base 预测为第 $i$ 类、性格模型预测为第 $j$ 类的样本数
- 总样本数: $N = \sum_{i=1}^K \sum_{j=1}^K c_{ij}$

从该矩阵计算三类 effect:

1. `effect_cramers_v`
2. `effect_tv`
3. `effect_js`

---

## 2) effect_cramers_v (Cramer's V)

### 2.1 二分类 (2x2)

对 $2 \times 2$ 矩阵：

$$
\mathbf{C} =
\begin{bmatrix}
c_{00} & c_{01} \\
c_{10} & c_{11}
\end{bmatrix}
$$

定义不一致方向的计数：

$$
n_{01} = c_{01}, \quad n_{10} = c_{10}, \quad n = n_{01} + n_{10}
$$

McNemar effect 统计量：

$$
\text{effect\_stat} = \frac{(n_{01} - n_{10})^2}{n}
$$

### 2.2 多分类 (KxK)

对 $K \times K$：

$$
\text{effect\_stat} = T
$$

其中 $T$ 为 Stuart-Maxwell 统计量（见显著性说明文档）。

### 2.3 Cramer's V 统一公式

$$
V = \sqrt{\frac{\text{effect\_stat}}{N (K - 1)}}
$$

### 2.4 解读

- 取值范围: $0 \sim 1$
- $V \approx 0$: 系统性差异很小
- $V$ 越大: 预测分布差异越明显

粗略参考：

- $0.0 \sim 0.1$: 极小
- $0.1 \sim 0.3$: 小
- $0.3 \sim 0.5$: 中
- $> 0.5$: 大

注意：如果 $n_{01} \approx n_{10}$，即使不一致样本很多，$V$ 也可能偏小。

---

## 3) effect_tv (Total Variation)

先得到边际分布：

$$
p_i = \frac{\sum_{j=1}^{K} c_{ij}}{N}, \quad
q_i = \frac{\sum_{j=1}^{K} c_{ji}}{N}
$$

TV 距离：

$$
\text{TV}(p, q) = \frac{1}{2} \sum_{i=1}^{K} \left| p_i - q_i \right|
$$

解读：

- 取值范围: $0 \sim 1$
- $\text{TV}=0$: 两套模型分布完全一致
- $\text{TV}=0.1$: 可理解为“整体上约 10% 的标签质量发生了迁移/偏移”

---

## 4) effect_js (Jensen–Shannon divergence)

定义：

$$
m = \frac{1}{2}(p + q)
$$

$$
\text{JS}(p, q) =
\frac{1}{2}\text{KL}(p \,\|\, m) +
\frac{1}{2}\text{KL}(q \,\|\, m)
$$

其中 KL 使用以 2 为底的对数：

$$
\text{KL}(p \,\|\, q) = \sum_{i=1}^{K} p_i \log_2 \frac{p_i}{q_i}
$$

解读：

- 取值范围: $0 \sim 1$
- $\text{JS}=0$: 分布完全一致
- 越大: 差异越明显

JS 相比 TV 更平滑，对小概率类别变化更敏感。

---

## 5) effect 与 p-value 的关系

- p-value: 差异是否“统计显著”
- effect: 差异“有多大”

样本量很大时，即使 effect 很小，p-value 也可能非常小。
建议先看显著性，再看 effect 是否具备实际意义。

---

## 6) 最小示例 (2x2)

给定：

$$
\mathbf{C} =
\begin{bmatrix}
70 & 9 \\
1 & 20
\end{bmatrix}
$$

步骤：

$$
n_{01} = 9,\quad n_{10} = 1,\quad n = 10
$$

$$
\text{effect\_stat} = \frac{(9 - 1)^2}{10} = 6.4
$$

$$
N = 100,\quad K = 2
$$

$$
V = \sqrt{\frac{6.4}{100 \cdot (2-1)}} = 0.253
$$

边际分布：

$$
p = \left[\frac{79}{100}, \frac{21}{100}\right],\quad
q = \left[\frac{71}{100}, \frac{29}{100}\right]
$$

TV：

$$
\text{TV} =
\frac{1}{2}\left(
\left|0.79 - 0.71\right| +
\left|0.21 - 0.29\right|
\right) = 0.08
$$

---

## 7) 在 CSV 里如何用

重点看三列：

- `effect_cramers_v`
- `effect_tv`
- `effect_js`

与 `p_value` 一起读取，即可回答“是否显著” + “差异多大”。
