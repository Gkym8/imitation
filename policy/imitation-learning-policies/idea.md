# 基于多头门控与延迟淘汰的长时程机器人记忆策略框架

## 1. 总体设计

本文设计一种面向长时程机器人策略学习的视觉记忆建模框架。核心思想是：历史记忆不仅需要根据当前状态进行检索，还应该根据其对未来策略学习的作用，学习每条历史记忆的重要性，并在记忆容量有限的情况下决定保留哪些历史信息。

与直接为每个历史 slot 维护一个独立的记忆权重不同，本文使用一个共享参数的 MLP Gate，为每条历史记忆在不同 Attention Head 上生成对应的重要性评分。Gate 一方面用于调整历史信息的 Attention 读取程度，另一方面用于 Memory Bank 满时的记忆淘汰。

同时，考虑到一条记忆在当前不重要，并不意味着它在未来一段时间内仍然不重要，本文引入延迟淘汰机制。被选中淘汰的记忆不立即删除，而是暂时保留为 Shadow Memory，以逐步衰减的权重继续参与未来 H 个 anchor 的记忆检索，使其能够继续从未来的 Policy Loss 和 Future Prediction Loss 中获得监督。

整体框架由五部分组成：

- **顺序记忆 Buffer**：按照固定时间间隔构建历史 memory slot，每个 slot 保存对应的历史视觉特征及必要的时间、状态信息；
- **Multi-Head Memory Gate**：使用共享参数的 MLP 为每条历史记忆生成多个 Gate，分别控制不同 Attention Head 对该记忆的读取程度；
- **Memory Retrieval**：将当前视觉与本体感受融合后作为 query，从历史 Memory Bank 中检索信息，构建当前决策记忆 h_t；
- **Shadow Memory 与延迟淘汰**：根据多个 Head 的 Gate 汇总分数选择淘汰对象，将其保留为低权重的 Shadow Memory，并在未来 H 个 anchor 后正式删除；
- **Future Prediction 与 Diffusion Policy**：使用当前决策记忆分别预测下一阶段的视觉特征和生成未来动作序列，通过两个任务的监督共同学习记忆检索与 Gate。 

整体数据流可以概括为：

$$  
\text{History}\rightarrow\text{Memory Bank} \xrightarrow{\text{Multi-Head Gate + Attention}}h_t  
$$

随后 $h_t$ 分别进入未来预测模块与 Diffusion 动作专家：

$$  
h_t\rightarrow \begin{cases} \text{Future Prediction}\rightarrow\mathcal L_{\text{pred}}\\[4pt] \text{Diffusion Policy}\rightarrow\mathcal L_{\text{diff}} \end{cases}  
$$

两个损失都通过 Attention 路径更新 Gate MLP，使其学习不同历史记忆对当前策略学习与未来状态预测的作用。未来预测作为训练时的辅助监督，其预测结果不重新写入 h_t；动作专家直接使用当前决策记忆生成动作。

## 2. 顺序记忆 Buffer

### 2.1 Slot 构建

历史轨迹按照固定时间间隔 L 构建时间有序的 anchor，每个 anchor 对应一个 memory slot：

$$  
\mathcal M_t=\{S_1,S_2,\dots,S_M\}.  
$$

第 i 个 slot 定义为：

$$  
S_i=\{m_i,t_i,r_i\},  
$$

其中：

- $m_i$：历史视觉特征；
- $t_i$：该 slot 对应的时间位置；
- $r_i$：该 slot 当前的状态，包括 Active 或 Shadow，以及 Shadow 剩余的保留时间。

每条 slot 保存对应时刻的视觉 token、时间位置及必要的保护状态，不额外保存历史动作序列。写入的是该时刻自身的观测特征，而不是已经检索并混入其他历史的决策记忆 h_t，避免将同一段历史反复写回 Memory Bank。

记忆按时间顺序写入，不根据 Gate 的大小决定是否创建新的 slot。当前观测先读取已有历史，完成当前决策后再作为新的 slot 写入，因此当前时刻不会把自己当作历史进行检索。

### 2.2 Active Memory 与 Shadow Memory

本文将 Memory Bank 区分为两种状态：

- **Active Memory**：正常参与 Attention 检索，并作为下一次淘汰决策的候选；
- **Shadow Memory**：已经被选中淘汰，但尚未物理删除的历史记忆。其 Attention 读取权重被大幅降低，并继续保留未来 H 个 anchor。

假设 Active Memory 的容量为 B=8。当有新的 slot 需要进入已经满载的 Active Bank 时，选择一个 Active Slot 转为 Shadow，新 slot 则进入 Active Bank。因此，Active Memory 始终最多保留 8 条，而 Shadow Memory 在后续 H 个 anchor 内暂时占用额外的物理空间。

如果每个 anchor 最多新建一条记忆，并且每次最多产生一条 Shadow，那么：

$$  
B_{\text{active}}=8,\qquad B_{\text{shadow}}\leq H,  
$$
$$  
\boxed{B_{\text{physical}}\leq8+H.}  
$$


这里的 H 表示延迟淘汰的 anchor 数量，而不是 Diffusion Policy 的动作预测长度。Active 与 Shadow 保存在同一个按时间排序的 Memory Bank 中，只通过状态和到期时间进行区分，不单独维护另一套 Shadow Bank。

## 3. Multi-Head Memory Gate 与当前记忆检索

### 3.1 Multi-Head Gate

原本每条历史记忆只对应一个 Gate，用于统一调控所有 Attention Head 的读取程度。但不同 Head 可能关注历史信息中的不同内容，因此本文将单个 Gate 扩展为 Multi-Head Gate。

同时，一条历史记忆的重要性与当前观测有关。因此，Gate 不仅输入历史 slot 自身的特征，还输入当前视觉与本体感受融合后的特征，使同一条历史记忆能够随当前状态的变化获得不同评分。

设当前视觉 token 与本体感受融合后得到 q_t，第 i 条历史记忆包含 N 个视觉 token：

$$
m_i=[m_{i,1},m_{i,2},\dots,m_{i,N}].
$$

分别对历史 slot 和当前观测的 token 取平均，再进行 LayerNorm：

$$
\bar m_i=\operatorname{LN}_{\text{mem}}\!\left(\operatorname{Mean}_{\text{token}}(m_i)\right),\qquad
\bar q_t=\operatorname{LN}_{\text{obs}}\!\left(\operatorname{Mean}_{\text{token}}(q_t)\right).
$$

将两个摘要在特征维拼接，使用一个共享参数的 MLP 生成各个 Head 的 Gate：

$$
\boxed{\mathbf g_{t,i}=\sigma\!\left(\operatorname{MLP}_{\text{gate}}\!\left[\bar m_i;\bar q_t\right]\right)
=[g_{t,i,1},g_{t,i,2},\dots,g_{t,i,N_h}].}
$$

其中 $N_h$ 表示 Attention Head 数量，$g_{t,i,h}\in(0,1)$ 表示在当前时刻 t，第 h 个 Head 对历史记忆 $S_i$ 的重要性评分。后文为了简化表达，将 $g_{t,i,h}$ 写为 $g_{i,h}$。

以特征维度 d=512、8 个 Head 为例，每个 Head 的特征维度为 64，Gate MLP 的结构为：

$$
1024\xrightarrow{\text{Linear}}128\xrightarrow{\text{GELU}}128\xrightarrow{\text{Linear}}8\xrightarrow{\text{Sigmoid}}\mathbf g_{t,i}.
$$

这里不是为每个 slot 或每个 Head 单独训练一个 MLP，而是所有历史 slot 共享同一个 MLP，由其一次输出 8 个 Gate。各个 Gate 独立使用 Sigmoid，不在不同 slot 之间做 Softmax。

每个 Gate 对应控制一个 Head 对该 slot 内全部 token 的读取。取平均只用于生成 Gate 的摘要，Memory Attention 仍然使用完整的历史 token。生成 Gate 时使用的 q_t 只包含当前视觉与本体感受，尚未混入历史读取结果。每次有新的当前观测时，重新计算所有保留 slot 的 Gate；同一次决策中的多层历史检索共享这组 Gate。

#### 多视角下的 Gate 输入

多视角图像分别经过共享视觉编码器，再在 token 维拼接，构成当前观测或同一条历史 slot。以 DINOv2 输出 768 维特征、每个相机保留 4 个 patch token 为例，单视角对应 4 个 token，三视角对应 12 个 token，经过投影后每个 token 的特征维度均为 512。

生成 Gate 时，对同一条 slot 内所有视角的 token 共同取平均，得到一个 512 维历史摘要；当前多视角 token 先通过 Attention 与本体感受融合，再共同取平均，得到一个 512 维当前摘要。

因此，单视角与三视角的 Gate MLP 输入维度相同，都是两个 512 维向量拼接后的 1024 维。三视角共同决定每条 slot 的 8 个 Gate，不为每个相机单独生成一组 Gate；完整的多视角 token 仍然保留给后续历史检索。

### 3.2 当前记忆检索

当前时刻得到视觉观测 o_t 与本体感受 p_t，首先通过视觉编码器得到：

$$  
x_t=E_{\text{vis}}(o_t).  
$$

随后融合视觉与本体感受信息：
$$  
q_t=F_{\text{obs}}(x_t,p_t).  
$$

  
将 q_t 作为当前 query，对 Memory Bank 中的历史 slot 进行多头 Attention 检索。

当前 query 与历史 slot 都可以包含多个 token。记 $q_{t,u}$ 为当前观测的第 u 个 query token，$m_{i,v}$ 为第 i 条历史记忆的第 v 个 token。第 h 个 Attention Head 的基础相似度为：

$$
s_{t,u,i,v}^{(h)}=\frac{(Q^{(h)}q_{t,u})^\top(K^{(h)}m_{i,v})}{\sqrt{d_h}}.
$$

历史时间位置信息进入 Key 的构建，使检索能够区分不同时间的历史记忆。

为了引入 Multi-Head Gate，本文不直接将 Gate 数值与基础相似度相加，而是在 Softmax 归一化之前，使用 Gate 调节每条历史记忆对应的指数权重。令 $\tilde g_{i,h}$ 表示第 i 条历史记忆在第 h 个 Head 上的有效 Gate：Active Memory 使用原始 $g_{i,h}$，Shadow Memory 则使用缩小后的 Gate，具体在第 5 节说明。

最终得到每个 Head 的 Attention 权重：

$$
\boxed{\alpha_{t,u,i,v}^{(h)}=
\frac{\tilde g_{i,h}\exp\!\left(s_{t,u,i,v}^{(h)}\right)}
{\sum_{j\in\mathcal M_t}\sum_w\tilde g_{j,h}\exp\!\left(s_{t,u,j,w}^{(h)}\right)}.}
$$

同一条 slot 在同一个 Head 上的 Gate 广播到该 slot 的所有 token，Softmax 则在所有有效历史 token 上共同归一化。

由于 $\tilde g_{i,h}>0$，上述乘性 Gate 形式在数学上等价于在 Softmax 前的 Attention Logit 上加入 $\log\tilde g_{i,h}$：

$$
\alpha_{t,u,i,v}^{(h)}=\operatorname{Softmax}_{i,v}\!\left(s_{t,u,i,v}^{(h)}+\log\tilde g_{i,h}\right).
$$

因此，Gate 在形式上直接调节 Softmax 归一化之前的指数权重，等价地发挥 Attention Bias 的作用。这里对应的加性 Bias 是 $\log\tilde g_{i,h}$，而不是直接将 $\tilde g_{i,h}$ 加到相似度上。

各个 Head 读取历史 Value：

$$
r_{t,u}^{(h)}=\sum_i\sum_v\alpha_{t,u,i,v}^{(h)}V^{(h)}m_{i,v}.
$$

将各个 Head 的结果拼接并映射后，得到每个当前 query token 对应的历史读取结果：

$$
r_{t,u}=W_O\operatorname{Concat}\left(r_{t,u}^{(1)},\dots,r_{t,u}^{(N_h)}\right).
$$

历史检索采用多层 Attention 与 FFN，并在每层保留 query 的残差。当前实现使用两层检索，输出 R_t 仍然具有与 q_t 相同的 token 数量和特征维度，包含当前观测及读取到的历史信息。

因此，模型中存在两种不同的历史选择机制：

**当前相关性**由 Query-Key Attention 决定，表示当前状态需要读取哪些历史信息；

**历史重要性**由 Multi-Head Gate 学习得到，表示该段历史信息对不同 Head 的读取和后续策略学习具有多大作用。

Gate 不再由预测误差显式赋值或手动更新，而是作为可学习网络，通过 Policy Loss 和 Future Prediction Loss 进行端到端优化。

### 3.3 当前观测与历史信息的融合

经过历史检索后，使用一个融合门控制当前观测 q_t 与历史检索输出 R_t 的组合：

$$
G_t=\sigma\!\left(W_F[q_t;R_t]+b_F\right),
$$

$$
\boxed{h_t=G_t\odot q_t+(1-G_t)\odot R_t.}
$$

这里的 G_t 为每个 token 的每个特征维度生成一个融合系数，形状与 q_t 相同。以特征维度 512 为例，融合门通过 Linear 将拼接后的 1024 维映射为 512 维，再经过 Sigmoid。最终 h_t 保留当前视觉 token 的数量，完整地作为 Diffusion Policy 的条件，不因为 Gate 打分时取了平均而压缩为单个 token。

该融合门与历史 slot 的 Multi-Head Gate 作用不同：Slot Gate 控制各个 Head 对每条历史的读取，融合门控制读取之后当前观测与历史增强表示的组合。

融合门通过 fusion_gate 选择是否启用。关闭时直接使用 h_t=R_t。由于 R_t 内部已经保留当前 query 的残差，关闭融合门后仍然包含当前观测与历史信息，而不是只使用历史读取结果。没有有效历史时，R_t=q_t，最终决策记忆也退化为当前观测表示。

## 4. 未来预测模块与 Diffusion Policy

### 4.1 Future Prediction

未来预测模块的目标不是显式建模完整未来轨迹，而是从当前决策记忆 h_t 中提取能够表征下一阶段状态变化的预测信息。

首先定义可学习的 Future Token：$p^F\in\mathbb R^d.$与当前决策记忆 h_t 和相应动作序列拼接：

$$  
X_t=[h_t;\,a_{t:t+k};\,p^F].  
$$
随后，将拼接后的 token 序列输入轻量级 Future Prediction Transformer：

$$  
\tilde X_t=T_{\text{pred}}(X_t).  
$$
取 Transformer 输出中 Future Token 对应的位置，并映射为预测的下一阶段视觉特征：

$$  
u_t^F=\tilde X_t[-1],\qquad \hat z_{t+1}=P_F(u_t^F).  
$$

对于下一个 anchor 中选取的真实代表帧 o_{t+1}，通过视觉编码器得到目标特征：$z_{t+1}=E_{\text{vis}}(o_{t+1}).$
使用余弦距离或者 MSE 损失。未来目标特征停止梯度，仅作为监督信号。动作条件使用当前 anchor 到下一 anchor 之间的真实示范动作，帮助预测该段动作之后的视觉变化。

未来预测分支与动作专家共享 h_t，但预测结果不反馈到动作专家，也不直接决定每条 slot 的 Gate。训练时的预测损失通过 h_t 和历史 Attention 更新共享 Gate MLP；推理时直接使用 h_t 生成动作，不需要执行未来预测分支。

该损失用于训练 Future Prediction Module，同时经由决策记忆 h_t 和 Memory Attention，更新 Multi-Head Gate。
因此，未来预测模块的目标是通过真实未来视觉特征监督，迫使当前决策记忆保留：

> 对下一阶段状态变化具有预测能力、并有助于未来决策的历史信息。

### 4.2 Diffusion Policy

经过加权检索后得到 h_t，并将其作为 Diffusion Action Expert 的条件，生成未来动作序列：

$$  
\pi_\theta(A_{t:t+T_a-1}\mid h_t),  
$$

其中 T_a 表示动作预测长度，区别于 Shadow Memory 的保留窗口 H。

Diffusion Policy 通过动作监督损失 $\mathcal L_{\text{diff}}$ 更新策略网络，同时也通过 Memory Attention 更新共享的 Gate MLP。

## 5. 基于 Gate 评分的记忆淘汰与 Shadow Memory

### 5.1 记忆淘汰评分

当 Active Memory Bank 达到容量上限 B=8，并需要写入新的历史 slot 时，使用当前 Gate MLP 对每条 Active Memory 的重要性进行评价。

对于第 i 条历史记忆，汇总其 N_h 个 Head 的原始 Gate：

$$  
\boxed{ I_i=\frac{1}{N_h}\sum_{h=1}^{N_h}g_{i,h} }.  
$$
随后选择汇总评分最低的 slot：

$$  
i^*=\arg\min_{i\in\mathcal M_t^{\text{active}}}I_i.  
$$
如果存在当前不允许淘汰的受保护 slot，则只在可淘汰的 Active Slots 中进行选择。评分来自当前这一次历史检索，不为每条 slot 单独维护持久化的权重。

保护机制包括首条记忆和最近记忆。keep_first 保护当前采样流的第一条 slot；keep_recent 将即将写入的新 slot 也计入最近记忆数量。例如 Bank 中已有 0～7，准备写入 8，当 keep_recent=2 时保护的是 7 和 8，不额外保护 6。首条记忆是否保护由 keep_first 单独决定。

训练和推理都先读取历史、得到当前 Gate，再选择淘汰对象并写入当前 slot，使用相同的保护范围和读写顺序。

该设计使同一个 Gate 同时承担两项功能：

- **Soft Memory Retrieval**：控制某条历史记忆在不同 Attention Head 中被读取的程度；
- **Hard Eviction Decision**：汇总多个 Head 的重要性评分，决定哪条历史记忆优先退出 Active Bank。

### 5.2 Shadow Memory 与低权重读取

传统的 Hard Eviction 在选中某条历史记忆后立即删除，使这条记忆无法再参与之后的 Policy Learning 和 Future Prediction。

但一条记忆当前评分较低，并不意味着它在未来一段时间内一定没有作用。因此，本文不立即删除被选中淘汰的 slot，而是将其状态由 Active 转为 Shadow，并继续保留未来 H 个 anchor。

本文根据 slot 当前的状态和进入 Shadow 后的读取次数设置有效 Gate：

$$
\boxed{\tilde g_{i,h}=\begin{cases}
g_{i,h},&S_i\text{ 为 Active},\\
\varepsilon\,10^{-a_i}g_{i,h},&S_i\text{ 为 Shadow}.
\end{cases}}
$$

其中 $\varepsilon$ 对应 shadow_gate_scale，$a_i$ 从 0 开始计数，表示该 slot 当前是第几次以 Shadow 状态被读取。

例如 shadow_horizon=3、shadow_gate_scale=0.01，则被选中淘汰的 slot 在之后三次历史读取中，分别将原始 Gate 乘以 0.01、0.001、0.0001，最后删除。

每次读取时都根据新的当前观测重新计算原始 Gate，再乘以该次的缩放系数，不在上一次已经缩小的 Gate 上继续累乘。该缩放作用于 Shadow 的每个 Head，并广播到该 slot 的所有 token。

因此，在第 3 节的乘性 Gate Attention 中，Shadow Memory 对应的未归一化权重变为 $\varepsilon\,10^{-a_i}g_{i,h}\exp(s)$，相当于在该条记忆的 Attention Logit 上额外加入 $\log\varepsilon-a_i\log10$ 的负向偏置。

这样，Shadow Memory 在 Softmax 归一化之前的指数权重逐步降低，使其通常只以很小的 Attention 权重继续参与记忆检索。由于 Softmax 还包含基础相似度和其他历史记忆的归一化，Shadow 最终的 Attention 权重不一定严格小于该次缩放系数。

需要注意，这里不是把 Gate 直接替换为固定常数，而是保留 MLP 原本预测的 $g_{i,h}$，再乘以缩放系数。因此 Shadow 的有效 Gate 仍然是一个可微函数，未来任务损失依然能够通过 Attention 回传到共享 Gate MLP。原始 Gate 与缩放后的有效 Gate 是两个不同的量，进入 Shadow 并不要求 MLP 的原始输出立即变小。

### 5.3 延迟物理删除

假设第 i 条历史记忆在 anchor t_e 被选中淘汰，则它进入 Shadow 状态，在之后的 H 个 anchor 中保持低权重参与 Attention：

$$  
t\in\{t_e+1,\dots,t_e+H\}.  
$$
在完成第 t_e+H 个 anchor 的训练或检索后，正式从 Memory Bank 中删除该 slot。

整个过程为：

$$  
\text{Active Memory} \xrightarrow{\text{最低 Gate 评分}} \text{Shadow Memory} \xrightarrow{\text{保留 }H\text{ 个 anchor}} \text{Physical Removal}.  
$$

Shadow Memory 不再参与之后的淘汰候选选择，也不会占用 8 个 Active Slots 的名额。每次新增记忆时，仍然只从 Active Bank 中选择下一条需要淘汰的记忆。

最低评分 slot 的选择和状态切换属于离散决策，不对 argmin 本身反向传播。Shadow 的作用是保留被选中记忆之后的可微读取路径，使后续任务损失仍可更新共享 Gate MLP，而不是为离散淘汰操作单独构造梯度。

### 5.4 可选的 Shadow 机制

Shadow 通过 shadow_window 选择是否启用，训练和推理使用同一设置。启用时，被选中的 slot 先转为 Shadow，按上述方式逐步衰减并延迟删除；关闭时直接删除最低评分的可淘汰 Active Slot，Memory Bank 只保留 Active 容量。shadow_horizon=0 时同样不产生延迟保留。

## 6. Gate Selectivity Regularization

Multi-Head Gate 由可学习的 MLP 产生，如果只使用任务损失进行训练，存在所有 Gate 输出趋于相同数值的可能，例如全部接近 1。此时 Gate 对不同历史记忆没有区分能力，无法真正发挥记忆选择的作用。

但本文不直接要求同一条记忆的 8 个 Head 必须产生不同的 Gate。因为某条历史记忆也可能同时对多个 Attention Head 都很重要。

真正希望约束的是：**对于同一个 Attention Head，不同历史记忆应当具有一定的 Gate 评分差异。**

假设当前 Active Bank 中有 B 条历史记忆，所有 Gate 构成矩阵：
$$  
G\in(0,1)^{B\times8}.  
$$
对于第 h 个 Attention Head，计算它在不同 Active Slots 上的 Gate 标准差：
$$  
\sigma_h=\operatorname{Std}_{i\in\mathcal M_t^{\text{active}}}(g_{i,h}).  
$$

当所有 slot 的 Gate 全部相同时，$\sigma_h=0$。因此，本文加入一个简单的选择性约束损失：
$$  
\boxed{ \mathcal L_{\text{gate}} =\frac{1}{N_h}\sum_{h=1}^{N_h} \left[\max(0,\tau-\sigma_h)\right]^2 }.  
$$


其中 $\tau$ 是预设的最低差异阈值，例如 0.1。

- 当某个 Head 对所有历史记忆的 Gate 都接近 1、接近 0，或者接近同一个常数时，标准差较小，会受到惩罚；
- 当某个 Head 已经能够对不同历史记忆产生一定的区分时，不再额外增大 Gate 差异；
- 不要求不同 Head 的 Gate 必须互不相同，而是允许多个 Head 同时关注同一条重要历史记忆。

该损失只对 Active Memory 的原始 Gate 计算，不将 Shadow 纳入统计。标准差使用各个 Active Slot 上的总体标准差。只有当前读取时 Active Bank 已满且至少有两条 Active Memory，才启用该项约束；未满时贡献为零，再对有效训练 anchor 取平均。
需要注意，该约束只用于防止 Gate 退化为常数输出，并不直接规定哪条历史记忆更重要。具体的重要性仍然通过任务损失进行学习。

## 7. 训练过程

现有训练仍然将演示轨迹组织为时间有序的 anchor 流，并按长度为 4 的小 chunk 进行训练。每个 chunk 额外读取一个 lookahead anchor，作为下一阶段视觉特征预测的监督目标。

首个 chunk 初始化 Memory Bank，后续 chunk 继续携带历史 slot、时间位置、保护状态和 Shadow 的剩余保留时间，而不是在每个 chunk 开始时重置 Memory Bank。

跨 chunk 时，保存历史特征和状态，但截断已有的历史计算图。在普通视觉 slot 路径中，后续 chunk 使用当前投影层重新映射保存的编码特征，并重新通过当前 Gate MLP 计算 Gate，使 Active 与 Shadow Memory 继续参与新 chunk 的前向计算与反向传播。

每个 anchor 都按照“读取已有历史—构建 h_t—使用当前 Gate 完成淘汰与写入”的顺序更新 Bank。Lookahead 仅作为当前 chunk 的未来监督目标，不提前写入 Bank；Padding 不写入，也不推进 Shadow 的保留时间。Shadow 的剩余时间只随有效 anchor 推进，不受 chunk 划分影响。

这里的未来窗口 H 按 anchor 数量计数，而不是按 chunk 数量计数。例如每个 chunk 为 4 个 anchor，当 H=8 时，被淘汰的 Shadow 可以跨越后续两个 chunk 接受任务损失监督。

与原本显式使用预测误差梯度更新记忆权重的方法不同，本方案不再通过独立的手动更新规则修改每条 slot 的 w_i，也不再需要由 Error Queue 为 Gate 直接赋予初始注意力偏置。Gate 由共享 MLP 输出，并通过现有的任务损失和选择性约束直接学习。

## 8. 总体训练目标

整体训练目标由三部分组成：

$$  
\boxed{ \mathcal L =\mathcal L_{\text{diff}} +\lambda_p\mathcal L_{\text{pred}} +\lambda_g\mathcal L_{\text{gate}} }.  
$$

其中：
- $\mathcal L_{\text{diff}}$：动作生成损失，负责学习最终机器人策略，同时监督记忆检索；
- $\mathcal L_{\text{pred}}$：未来视觉特征预测损失，迫使决策记忆保留能够解释下一阶段状态变化的信息；
- $\mathcal L_{\text{gate}}$：Gate 选择性约束，鼓励同一 Head 对不同历史记忆产生具有区分性的评分，避免 Gate 退化为常数输出。

Active Memory 与 Shadow Memory 使用相同的任务损失和同一个 Gate MLP。不同之处仅在于 Shadow Memory 的有效 Gate 被乘以逐步衰减的系数 $\varepsilon\,10^{-a_i}$，并在未来 H 个 anchor 后被物理删除。

因此，不需要单独设计针对 Shadow 的未来监督损失，而是通过延迟删除保留已有的任务监督路径。

## 9. 推理阶段

推理阶段沿用与训练阶段相同的顺序记忆构建、Multi-Head Gate、Attention 检索。
当前时刻：

$$  
(o_t,p_t)\rightarrow q_t \rightarrow\mathcal M_t \xrightarrow{\text{Multi-Head Gate + Attention}}h_t.  
$$
随后：
$$  
h_t\rightarrow\text{Diffusion Policy} \rightarrow A_{t:t+T_a-1}.  
$$
历史检索与 Gate 在每次新的观测到来时计算一次，得到的 h_t 在本次 Diffusion 的多个去噪步骤中重复使用，不在每个去噪步骤重新检索历史。推理不运行未来预测模块。

动作序列生成后，使用本次读取产生的原始 Gate 汇总分数处理已满的 Active Bank：仅在未受保护的 Active Slots 中选择最低评分对象；启用 Shadow 时转为 Shadow，关闭时直接删除。已经存在的 Shadow 在完成最后一次有效读取后删除，随后将当前观测作为新的 Active Slot 写入。

执行一段动作并获取下一次观测后，重复上述过程。每条 episode 独立维护自己的 Memory Bank，在 episode 重置时清空，不跨 episode 共享历史。
