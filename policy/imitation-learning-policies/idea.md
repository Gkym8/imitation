# 基于预测误差加权的长时程机器人记忆策略框架

## 1. 总体设计

本文设计一种面向长时程机器人策略学习的记忆建模框架。核心思想是：记忆不仅用于存储历史信息，还应根据历史状态对未来行为的重要程度动态调整其读取权重。

整体框架由四部分组成：

- **顺序记忆 Buffer**：以固定时间间隔将历史轨迹划分为若干 slot，每个 slot 保存一段动作轨迹以及代表性历史图像特征；
- **记忆检索与决策表征构建**：当前视觉观测与本体感受作为 query，从历史 Buffer 中检索相关信息，得到当前决策记忆 $h_t$；
- **未来预测与动态权重更新**：利用 $h_t$ 预测下一阶段未来视觉表征，通过预测误差衡量未来状态相对于当前记忆的“不可预期程度”，进而更新对应历史 slot 的记忆权重；
- **Diffusion 动作专家**：以 $h_t$ 作为条件生成未来动作序列

整体数据流可以概括为： $$ \text{History} \rightarrow \text{Memory Buffer} \rightarrow \text{Memory Retrieval} \rightarrow h_t $$

随后 $h_t$ 分别进入未来预测模块与 Diffusion 动作专家： $$ h_t \rightarrow \begin{cases} \text{Future Prediction} \rightarrow \text{Prediction Error} \rightarrow \text{Memory Weight Update}\\[4pt] \text{Diffusion Policy} \rightarrow A_{t:t+H-1} \end{cases} $$


## 2. 顺序记忆 Buffer

### 2.1 Slot 构建

历史轨迹按照固定长度 $L$ 顺序划分为 $M$ 个 memory slot： $\mathcal M_t = { S_1,S_2,\dots,S_M }.$

第 $i$ 个 slot 定义为： $$ S_i = { A_i,, v_i,, w_i }, $$ 其中：

- $A_i$：该时间段内连续的历史动作轨迹；
- $v_i$：该时间段对应的代表性时刻视觉特征；
- $w_i$：该 slot 图像当前的记忆重要性权重。

例如： $$ \begin{aligned} S_1&: {a_0,\dots,a_{L-1},v_1},\ ; S_2: {a_L,\dots,a_{2L-1},v_2}. \end{aligned} $$

视觉帧可以统一开始时刻

该设计不根据预测误差决定是否写入记忆， 即预测误差只改变历史信息未来被读取时的重要程度，而不改变历史数据本身的时间结构。

## 3. 当前记忆检索与决策表征

当前时刻得到视觉观测 $o_t$ 与本体感受 $p_t$。首先通过视觉编码器得到： $x_t=E_{\text{vis}}(o_t).$ 然后融合视觉和本体信息： $q_t = F_{\text{obs}}(x_t,p_t).$

将 $q_t$ 作为当前 query，对 Memory Buffer 中的历史 slot 进行检索。

基础 attention 为： $$ s_i = \frac{ (Qq_t)^\top(Km_i) }{ \sqrt d }. $$

为了引入预测误差所得到的历史重要性，使用 memory weight $w_i$ 对 attention score 进行调制： $\tilde s_i = s_i + w_i.$

最终： $$ \begin{aligned} \alpha_i &= \operatorname{Softmax}(\tilde s_i),\ h_t= \sum_i \alpha_i Vm_i. \end{aligned} $$

因此，模型中实际上存在两种不同的历史选择机制：

**当前相关性** 由 attention 决定： $q_t \leftrightarrow m_i.$ 表示当前状态需要查询哪些历史信息？

**历史重要性** 由预测误差产生： $w_i.$ 表示这段历史在过去是否对应重要、不可预期的状态变化？

## 4. 未来预测模块

### 4.1 Future Prediction

未来预测模块的目标不是显式建模完整未来轨迹，而是从当前决策记忆 $h_t$ 中提取能够表征下一阶段状态变化的预测信息。

首先定义一个可学习的 Future Token：$p^F \in \mathbb{R}^{d}.$

该 token 不直接表示未来图像特征，而是作为一个未来预测占位符，与当前决策记忆 $h_t$ 拼接：$X_t = [h_t;,p^F].$

随后，将拼接后的 token 序列输入一个轻量级 Future Prediction Transformer：$T_{\text{pred}}(X_t).$

取 Transformer 输出中 Future Token 对应的位置：$\tilde X_t[-1].$

其中，$u_t^F$ 表示在当前决策记忆 $h_t$ 条件下得到的未来预测隐表示。

### 4.2 未来视觉特征监督

对于下一个 memory slot 中选取的真实代表帧 $o_{t+1}$，使用视觉编码器 $E_{\text{vis}}$ 提取其真实视觉特征：$E_{\text{vis}}(o_{t+1}).$

训练过程中，使用真实未来视觉特征监督预测结果。本文采用余弦距离作为未来预测误差：

$$1-  
\frac{  
\hat z_{t+1}^{\top}  
(z_{t+1})  
}{  
|\hat z_{t+1}|_2  
|  
(z_{t+1})  
|_2  
}.  
$$
定义当前预测误差为：$\mathcal L_{\text{pred}}.$

该误差除了用于训练 Future Prediction Module 外，还将在后续作为记忆动态调节的反馈信号，通过计算预测误差关于 memory attention bias $w_i$ 的梯度：
$$\frac{\partial e_t}{\partial w_{i,t}},$$

衡量第 $i$ 个历史 memory slot 对当前未来预测的贡献，并进一步动态调整其检索权重。

因此，未来预测模块的目标并不是构建能够生成完整未来场景的世界模型，而是通过未来视觉特征监督，迫使当前决策记忆 $h_t$ 保留：

> 对下一阶段状态变化具有预测能力、并真正有助于未来决策的历史信息。

整个过程可以概括为：

$$  
h_t  
\rightarrow  
[h_t;p^F]  
\rightarrow  
T_{\text{pred}}  
\rightarrow  
u_t^F  
\rightarrow  
\hat z_{t+1}  
\rightarrow  
e_t.  
$$

其中，$e_t$ 一方面作为未来预测监督信号，另一方面作为后续预测误差驱动记忆权重更新的基础。

## 5. 基于预测误差的记忆权重更新

未来预测不仅作为辅助训练任务，同时用于判断某个历史状态的重要程度。假设当前通过 slot $S_i$ 及之前的信息，预测下一阶段对应视觉特征： $\hat z_{i+1}.$

当真实的下一 slot $S_{i+1}$ 到来后，得到真实视觉特征： $z_{i+1}.$

计算预测误差： $e_{i+1} = D \left( \hat z_{i+1}, z_{i+1} \right).$

需要注意，$e_{i+1}$ 描述的是：

> 在看到 $S_{i+1}$ 之前，根据过去历史对它进行预测时产生了多大的 surprise。

因此该误差应赋给： ${S_{i+1}}$ 而不是 $S_i$。

## 6. Error Queue 与初始记忆权重

直接使用绝对预测误差（prediction error）衡量记忆重要性存在一个问题：随着未来预测模块逐渐收敛，预测误差的整体尺度可能不断减小。因此，不能简单地令： $$ w_t=e_t $$ 本文维护一个长度为 $N_e$ 的滑动误差队列（Error Queue）： $$ \mathcal E_t = {e_{t-N_e},\dots,e_{t-1}} $$ 其中存储近期未来预测产生的预测误差。

对于当前预测误差 $e_t$，首先计算误差队列中误差的均值与标准差： $$ \mu_Q = \frac{1}{N_e} \sum_{e_j\in\mathcal E_t} e_j,\qquad \sigma_Q = \sqrt{ \frac{1}{N_e} \sum_{e_j\in\mathcal E_t} (e_j-\mu_Q)^2 } $$

随后使用 Z‑分数衡量当前预测误差相对于近期预测误差分布的位置： $$ z_t = \frac{ e_t-\mu_Q }{ \sigma_Q+\epsilon } $$

其中：

- $z_t>0$ 表示当前状态比近期平均状态更难预测；
- $z_t<0$ 表示当前状态比近期平均状态更容易预测；
- $z_t\approx0$ 表示当前预测误差接近近期平均水平。

由于 Z‑分数描述的是预测误差的相对位置，因此即使随着训练进行所有预测误差的绝对值逐渐减小，只要不同状态之间仍然存在相对预测难度差异，该指标仍然可以稳定地区分其相对重要性。

随后，将标准化后的预测误差映射为对应记忆槽（memory slot）的初始注意力偏置： $$ w_t^{(0)} = w_{\min} + (w_{\max}-w_{\min}) \cdot \sigma \left( z_t\right) $$ 其中 $\sigma(\cdot)$ 表示 Sigmoid 函数

由于 $w_t$ 将直接作为加性偏置（additive bias）加入注意力对数（attention logits），因此建议使权重范围以 $0$ 为中心，即： $$ w_{\min}=-\lambda, \qquad w_{\max}=+\lambda $$ 此时可以进一步写为： $$ w_t^{(0)} = \lambda \left[ 2\sigma \left(z_t \right)-1 \right] $$

因此：

- $z_t=0 \Rightarrow w_t^{(0)}=0$，表示预测难度处于近期平均水平时，不对原始注意力得分施加额外偏置；
- $z_t>0 \Rightarrow w_t^{(0)}>0$，表示相对难预测的状态获得更高的初始记忆优先级；
- $z_t<0 \Rightarrow w_t^{(0)}<0$，表示相对容易预测的状态受到一定程度的抑制。

因此，误差队列提供的是一种基于历史预测惊喜度（prediction surprise）的**记忆重要性先验（Memory Importance Prior）**。

## 7. Prediction‑Error‑Guided Memory Weight Update

误差队列仅7决定记忆槽被建立时的初始权重$w_i^{(0)}$，然而，一个历史状态在产生时具有较大的预测误差，并不意味着它在之后所有决策时刻都同样重要。因此，在记忆检索（Memory Retrieval）过程中，本文进一步利用当前未来预测误差对已有记忆权重进行动态修正。

对于当前查询 $q_t$ 与第 $i$ 个记忆槽，其原始注意力相似度为： $$ s_{i,t} = \frac{ q_t^\top k_i }{ \sqrt d } $$

将动态记忆权重 $w_{i,t}$ 直接作为加性注意力偏置： $$ \tilde s_{i,t} = s_{i,t}+w_{i,t} $$

随后计算注意力： $$ \alpha_{i,t} = \operatorname{softmax} ( \tilde s_{i,t} ) $$

并得到当前决策记忆： $$ h_t = \sum_i \alpha_{i,t}v_i $$

未来预测模块根据 $h_t$ 预测下一记忆槽的代表帧视觉特征，并得到当前预测误差： $$ e_t = \mathcal L_{\text{pred}} $$

由于记忆权重 $w_{i,t}$ 参与了当前预测误差的完整前向计算过程： $$ w_{i,t} \rightarrow \alpha_{i,t} \rightarrow h_t \rightarrow \hat z_{t+1} \rightarrow e_t $$ 因此可以计算预测误差关于每个记忆权重的梯度： $$ g_{i,t} = \frac{ \partial e_t }{ \partial w_{i,t} } $$


该梯度反映：如果进一步提高第 $i$ 个记忆槽的检索权重，当前未来预测误差将如何变化。

- 当 $g_{i,t}<0$，意味着增加 $w_{i,t}$ 可以降低预测误差，因此该记忆槽对当前未来预测具有正向贡献，应提高其检索权重。
- 反之，当 $g_{i,t}>0$，意味着增加该记忆槽的权重会进一步增大预测误差，因此应降低其检索权重。

因此采用梯度下降形式进行动态更新： $$ w_{i,t}^{\text{new}} = w_{i,t} - \eta_w \operatorname{sg} \left( g_{i,t} \right) $$ 其中 $\eta_w$ 为记忆权重更新步长，$\operatorname{sg}(\cdot)$ 表示**停止梯度（Stop Gradient）**。

需要强调，$w_i$ 不是模型中的可学习参数，也不由网络优化器更新，而是维护在模型参数之外的动态记忆状态。每次进行未来预测时，仅临时令其参与计算图，从而获得 $\frac{\partial e_t}{\partial w_i}$，随后立即断开梯度（detach），并通过上述显式更新规则修改 $w_i$。

为了避免单次预测误差产生过大的权重变化，可以进一步对更新结果进行截断： $$ w_{i,t}^{\text{new}} = \operatorname{clip} \left( w_{i,t} - \eta_w \operatorname{sg}(g_{i,t}), w_{\min}, w_{\max} \right) $$

因此，本文中的记忆权重实际包含两个连续阶段： $$ \boxed{ \text{Historical Prediction Error} \rightarrow \text{Initial Memory Weight} } $$ 以及： $$ \boxed{ \text{Current Prediction Error Gradient} \rightarrow \text{Dynamic Weight Correction} } $$

其中，误差队列回答的是：**该状态在被记录时，相对于近期历史而言有多难预测？** 而预测误差梯度回答的是：**该历史记忆在当前决策状态下，对正确预测未来究竟有多大帮助？**

最终记忆检索为： $$ \boxed{ \alpha_{i,t} = \operatorname{softmax} \left( \frac{ q_t^\top k_i }{ \sqrt d } + w_{i,t} \right) } $$

从而形成完整闭环： $$ \boxed{ \text{Predict} \rightarrow \text{Error} \rightarrow \text{Initialize / Correct Weight} \rightarrow \text{Memory Retrieval} \rightarrow \text{Predict} } $$

该机制使记忆权重不再是固定的历史重要性评分，而是能够随着后续预测结果不断动态调整的外部记忆状态。


## 8. 最终 Diffusion 策略学习

经过加权检索后得到：$h_t$。

$h_t$ 一方面进入 Future Prediction Module，另一方面直接作为 Diffusion Action Expert 的条件。

最终策略为： $$ \pi_\theta( A_{t:t+H-1} \mid h_t ). $$

其中 $h_t$ 已经通过历史 attention 同时融合： $\text{Current Observation Relevance}$ 以及 $\text{Historical Predictive Importance}$.

因此最终动作生成并不是依赖均匀的历史信息，而是重点利用：

> 当前任务相关，同时在过去发生时具有较高预测 surprise 的历史片段。

## 9. 总体训练目标

整体训练损失保持为三个主要组成部分： $$ \boxed{ \mathcal L = \mathcal L_{\text{diff}} + \lambda_p\mathcal L_{\text{pred}}  } $$

- 动作生成损失 $\mathcal L_{\text{diff}}$ 负责学习最终机器人策略。
- 未来预测损失 $\mathcal L_{\text{pred}}$ 迫使决策记忆保存能够解释未来状态的信息。

## 10. 推理阶段

历史 slot 按照与训练阶段相同的固定时间顺序持续构建。

当前时刻： $$ (o_t,p_t) \rightarrow q_t \rightarrow \mathcal M_t \rightarrow h_t. $$

然后： $$ h_t\rightarrow \text{Diffusion Policy}\rightarrow A_{t:t+H-1}. $$

未来预测模块仍可以在线预测下一阶段表示： $$ h_t\rightarrow \hat z_{t+k}. $$

当未来真实观测真正到达后，再计算 prediction error，并更新该新历史 slot 的权重。

因此整个推理过程是严格因果的： $$ \boxed{ \text{Prediction} \rightarrow \text{Future Observation Arrives} \rightarrow \text{Error} \rightarrow \text{Memory Weight Update} } $$ 不会在动作决策时使用尚未发生的真实未来信息。

 

