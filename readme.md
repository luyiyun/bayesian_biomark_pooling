# Hierarchical Bayesian Models for pooled and calibrated biomarker data

## Model

$$W\leftarrow X\rightarrow Y\leftarrow Z$$

## Simulation

### Scenario 1

from "Statistical methods for biomarker data pooled from multiple nested case-control studies"。

模型可以简单表示为：$W\rightarrow X\rightarrow Y$

1. 生成$`X_{si}`$，$`W_{si}`$，需要设置参数$`a_s, b_s, \mu_x, \sigma^2_x, \sigma^2_{ws}`$；

$$
\left(\begin{array}{c}X_{s i} \\ W_{s i} \\ e_{s i}\end{array}\right) \sim \text{MVN}\left(\left(\begin{array}{c}\mu_{x} \\ \left(\mu_{x}-a_{s}\right) / b_{s} \\ 0\end{array}\right),\left(\begin{array}{ccc}\sigma_{x}^{2} & b_{s} \sigma_{w s}^{2} & \sigma_{x}^{2}-b_{s}^{2} \sigma_{w s}^{2} \\ \cdot & \sigma_{w s}^{2} & 0 \\ \cdot & \cdot & \sigma_{x}^{2}-b_{s}^{2} \sigma_{w s}^{2}\end{array}\right)\right)
$$

这等价于以下的过程：

$$
\begin{array}{}
W_{si}\sim\mathcal{N}(\mu_{ws}, \sigma^2_{ws}) \\
e_{si}\sim\mathcal{N}(0,\sigma^2_{es}) \\
X_{si}=a_s+b_sW_{si}+e_{si}
\end{array}
$$

其中$`\mu_{ws}=(\mu_x-a_s)/b_s`$，$`\sigma^2_{es}=\sigma^2_x-b_s^2\sigma^2_{ws}`$。
但是需要注意到，**如果我们通过设置$`\sigma^2_{ws}`$来计算$`\sigma^2_{es}`$，容易算出负值。所以我们这里是来设置$`\sigma^2_{es}`$，进而反推出**$`\sigma^2_{ws}`$。

2. 生成$`Y_{si}`$，需要参数$`\beta_{0s}`$，$`\beta_x`$。

$$
\text{logit}(P(Y_{si}=1|X_{si}))=\beta_{0s}+\beta_{x}X_{si}
$$

3. 每个$`s`$下，移除部分$`X_{si}`$。

因为相关性没有方向，所以我们也可以转换成X->W的方向。此时，我们可以得到下面的结果：

$$
\begin{array}{}
W_{si}|X_{si}\sim\mathcal{N}(A+BX_{si}, C) \\
A=(\sigma_{es}^2\mu_{w}-\sigma_{ws}^2a_sb_s)/(b_s^2\sigma_{ws}^2+\sigma_{es}^2) \\
B=(b_s\sigma_{ws})/(b_s^2\sigma_{ws}^2+\sigma_{es}^2) \\
C=(\sigma_{ws}^2\sigma_{es}^2)(b_s^2\sigma_{ws}^2+\sigma_{es}^2)
\end{array}
$$


### Scenario 2

模型可以简单表示为：$`W\leftarrow X\rightarrow Y`$，**更加符合我们模型的假设**。

1. 生成$`X_{si}`$，$`W_{si}`$，需要设置参数$`a_s, b_s, \mu_x, \sigma^2_x, \sigma^2_{es}`$；

$$
\begin{array}{}
X_{si}\sim\mathcal{N}(\mu_{x}, \sigma^2_{x}) \\
e_{si}\sim\mathcal{N}(0,\sigma^2_{es}) \\
W_{si}=a_s+b_sX_{si}+e_{si}
\end{array}
$$

可以计算得到三者服从以下的MVN：

$$
\left(\begin{array}{c}X_{s i} \\ W_{s i} \\ e_{s i}\end{array}\right)
\sim \text{MVN}\left(
    \left(\begin{array}{c}\mu_{x} \\ a_s+b_s\mu_x \\ 0\end{array}\right),
    \left(\begin{array}{ccc}
        \sigma_{x}^{2} & b_{s} \sigma_{x}^{2} & 0 \\
        \cdot & b_s^2\sigma_x^2+\sigma_{es}^2 & \sigma_{es}^2 \\
        \cdot & \cdot & \sigma_{es}^2
    \end{array}\right)
\right)
$$

2. 生成$`Y_{si}`$，需要参数$`\beta_{0s}`$，$`\beta_x`$。

$$
\text{logit}(P(Y_{si}=1|X_{si}))=\beta_{0s}+\beta_{x}X_{si}
$$

4. 每个$`s`$下，移除部分$`X_{si}`$。

### Experiments
1. 不同prevalence, 不同OR值下的结果？
2. 不同的$`\sigma_{es}^2`$大小，如果较大，可能会不符合Sloan' method的假设？
3. $`X_{si}`$可知的样本比例的影响，特别是如果存在一些studies($`s`$)没有$`X_{si}`$？
4. 可能存在一种情况：一些studies只有$`X_{si}`$，另一些studies只有$`W_{si}`$？
5. 如果存在协变量$`Z_{si}`$来影响$`Y_{si}`$，这个协变量同样可能影响$`X_{si}`$或$`W_{si}`$?
6. 其他的常用情况：不同的OR值($`\beta_x`$)，不同studies样本量的影响，等等。
7. 不同的几个先验设置的影响？
8. 使用ADVI和使用MCMC的差别？
