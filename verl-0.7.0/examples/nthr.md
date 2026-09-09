# NTHR for GRPO — Implementation-Oriented Method Note

This note contains only the core method and formulas needed to implement **Negative Token Hidden Reward (NTHR)** on top of an existing GRPO trainer.

The VERL example implementation, launch command, supported scope, and tests are
in [`examples/nthr_trainer`](nthr_trainer/README.md).

The key idea is simple:

> In an incorrect rollout, do **not** penalize every token equally.  
> If a token looks strongly aligned with the hidden/prediction-error geometry of the correct rollouts from the same prompt, reduce its negative advantage.

---

## 1. Setup

For one prompt \(x\), sample \(G\) responses

\[
\{y_i\}_{i=1}^{G} \sim \pi_{\text{old}}(\cdot \mid x)
\]

with rewards \(r_i\).

For binary correctness rewards,

\[
r_i \in \{0,1\}.
\]

Split the group into

\[
\mathcal P = \{i:r_i=1\},
\qquad
\mathcal N = \{j:r_j=0\}.
\]

Let

\[
N^+ = |\mathcal P|,
\qquad
N^- = |\mathcal N|,
\qquad
p = \frac{N^+}{G}.
\]

NTHR is meaningful only for a **mixed group**:

\[
0 < N^+ < G.
\]

If all responses are correct or all are incorrect, there is no positive-vs-negative comparison. In practice, use standard GRPO behavior or skip the zero-advantage group.

---

## 2. Standard GRPO advantage

For normal GRPO, compute a response-level normalized advantage

\[
\hat A_i
=
\frac{r_i-\mu_r}{\sigma_r+\epsilon_A}.
\]

For binary reward, approximately

\[
\mu_r = p,
\qquad
\sigma_r = \sqrt{p(1-p)}.
\]

Therefore,

\[
\hat A_i
=
\begin{cases}
\sqrt{\frac{1-p}{p}}, & r_i=1, \\[6pt]
-\sqrt{\frac{p}{1-p}}, & r_i=0.
\end{cases}
\]

Normally every completion token in response \(i\) receives the same \(\hat A_i\).

NTHR changes this **only for selected tokens in incorrect responses**.

---

## 3. Token quantities needed by NTHR

For each generated response token \(y_{i,k}\), save from the rollout / old-policy forward pass:

- final-layer hidden state

\[
h_{i,k}
=
h_{x,y_{i,<k}}
\in \mathbb R^d,
\]

- next-token probability vector

\[
q_{i,k}
=
\pi_{\text{old}}(\cdot \mid x,y_{i,<k}),
\]

- sampled token id \(y_{i,k}\).

Define the token prediction-error vector

\[
g_{i,k}
=
e_{y_{i,k}} - q_{i,k},
\]

where \(e_{y_{i,k}}\) is the one-hot vector of the generated token.

The paper's token-level prediction-error similarity is

\[
\alpha_{(i,k),(j,k')}
=
\left\langle
g_{i,k},
g_{j,k'}
\right\rangle.
\]

For a positive token and a negative token,

\[
\alpha^-_{k,k'}
=
\left\langle
e_{y^+_{i,k}}
-
\pi_{\text{old}}(\cdot\mid x,y^+_{i,<k}),
\;
e_{y^-_{j,k'}}
-
\pi_{\text{old}}(\cdot\mid x,y^-_{j,<k'})
\right\rangle.
\]

---

## 4. NTHR score for an incorrect token

For token \(k'\) of incorrect response \(y^-_j\), define

\[
s^-_{j,k'}
=
\sum_{i=1}^{N^+}
\sum_{k=1}^{|y_i^+|}
\alpha^-_{k,k'}
\left\langle
h^+_{i,k},
h^-_{j,k'}
\right\rangle.
\]

Equivalently,

\[
\boxed{
s^-_{j,k'}
=
\sum_{i=1}^{N^+}
\sum_{k=1}^{|y_i^+|}
\left\langle g^+_{i,k},g^-_{j,k'}\right\rangle
\left\langle h^+_{i,k},h^-_{j,k'}\right\rangle
}
\]

A large positive \(s^-_{j,k'}\) means that penalizing this negative token is predicted to strongly interfere with the correct responses.

Those are the tokens whose negative gradient should be weakened.

---

## 5. Efficient matrix form

Do **not** explicitly compare every positive token with every negative token.

Flatten all positive completion tokens for the current prompt.

Let

\[
G^+ \in \mathbb R^{T_+ \times V_*},
\qquad
H^+ \in \mathbb R^{T_+ \times d},
\]

where

- \(T_+\) = total number of positive completion tokens,
- \(V_*\) = vocabulary subset used for NTHR.

Construct

\[
\boxed{
M^+ = (G^+)^\top H^+
}
\]

so that

\[
M^+ \in \mathbb R^{V_* \times d}.
\]

For one token with prediction-error vector \(g_t\) and hidden state \(h_t\),

\[
\boxed{
s_t = g_t^\top M^+ h_t
}
\]

which exactly equals the double sum above.

For all negative tokens at once,

\[
\boxed{
s^- =
\operatorname{rowsum}
\left[
(G^- M^+) \odot H^-
\right]
}
\]

with

\[
G^- \in \mathbb R^{T_- \times V_*},
\qquad
H^- \in \mathbb R^{T_- \times d}.
\]

In PyTorch-like notation:

```python
M_pos = G_pos.T @ H_pos                  # [V*, d]
s_neg = ((G_neg @ M_pos) * H_neg).sum(-1)  # [T_neg]
```

This is the most useful form for implementation.

---

## 6. Restricting the vocabulary

The full prediction-error vector has dimension \(|V|\), which is expensive.

For each prompt, define

\[
V_x^*
=
\{\text{unique token ids appearing in the sampled responses for }x\}.
\]

Then compute \(g_{i,k}\) only on these vocabulary columns.

Thus

\[
|V_x^*| \ll |V|.
\]

For implementation, slice the model probabilities to \(V_x^*\):

\[
g_{i,k}^{(*)}
=
e_{y_{i,k}}^{(*)}
-
q_{i,k}[V_x^*].
\]

Do **not** renormalize the sliced probabilities; this is an approximation to the original full-vocabulary vector obtained by dropping unused coordinates.

---

## 7. Positive-response anchor score

NTHR needs a threshold indicating how large a token influence is relative to interactions among correct responses themselves.

Using the same positive aggregate matrix \(M^+\), compute a score for every positive token:

\[
s^+_{i,k}
=
(g^+_{i,k})^\top
M^+
h^+_{i,k}.
\]

For each positive response \(i\), average over its completion tokens:

\[
\boxed{
\bar s_i^+
=
\frac{1}{|y_i^+|}
\sum_{k=1}^{|y_i^+|}
s^+_{i,k}
}
\]

which is equivalent to

\[
\bar s_{i'}^+
=
\frac{1}{|y_{i'}^+|}
\sum_{k''=1}^{|y_{i'}^+|}
\sum_{i=1}^{N^+}
\sum_{k=1}^{|y_i^+|}
\alpha^+_{k,k''}
\left\langle
h^+_{i,k},
h^+_{i',k''}
\right\rangle.
\]

Then define the threshold

\[
\boxed{
\tau
=
\beta
\min_{i\in\mathcal P}
\bar s_i^+
}
\]

The paper uses

\[
\boxed{\beta=1}
\]

as its main setting.

---

## 8. Select tokens in incorrect responses

For each incorrect token,

\[
m_{j,k'}
=
\mathbbm 1
\left[
s^-_{j,k'} > \tau
\right].
\]

Equivalently,

\[
\boxed{
\mathcal V_j^-
=
\left\{
y^-_{j,k'} :
s^-_{j,k'} > \tau
\right\}
}
\]

Only these high-NTHR negative tokens receive weaker penalization.

---

## 9. NTHR-modified token advantage

For a normal GRPO token,

\[
\hat A_{i,k} = \hat A_i.
\]

For a selected token in an incorrect response, replace it with

\[
\boxed{
\hat A^{\text{NTHR}}_{j,k'}
=
\eta \hat A_j
}
\]

where

\[
0 \le \eta < 1.
\]

Therefore the full rule is

\[
\boxed{
\hat A^{\text{NTHR}}_{i,k}
=
\begin{cases}
\eta \hat A_i,
&
r_i=0
\text{ and }
s^-_{i,k}>\tau,
\\[4pt]
\hat A_i,
&
\text{otherwise}.
\end{cases}
}
\]

Because \(\hat A_i<0\) for an incorrect response, multiplying by \(\eta<1\) makes its gradient less negative.

The main paper setting is

\[
\boxed{
\eta
=
2|0.5-p|
}
\]

where

\[
p = \frac{N^+}{G}.
\]

Examples:

\[
p=0.5 \Rightarrow \eta=0,
\]

\[
p=0.25 \Rightarrow \eta=0.5,
\]

\[
p=0.1 \Rightarrow \eta=0.8.
\]

So NTHR most strongly suppresses selected negative-token penalties when the rollout group contains a balanced mixture of successes and failures.

---

## 10. Plug into the normal GRPO objective

Let the token probability ratio be

\[
\rho_{i,k}(\theta)
=
\frac{
\pi_\theta(y_{i,k}\mid x,y_{i,<k})
}{
\pi_{\text{old}}(y_{i,k}\mid x,y_{i,<k})
}.
\]

Use the NTHR-modified token advantage inside the same clipped GRPO/PPO surrogate:

\[
\boxed{
L_{\text{policy}}
=
-
\frac{1}{T}
\sum_{i,k}
\min
\left(
\rho_{i,k}\hat A^{\text{NTHR}}_{i,k},
\;
\operatorname{clip}
(\rho_{i,k},1-\epsilon,1+\epsilon)
\hat A^{\text{NTHR}}_{i,k}
\right)
}
\]

plus whatever KL regularization your GRPO implementation already uses.

**Nothing else in the optimizer needs to change.**

---

## 11. Vectorized implementation recipe

For every prompt group:

1. Generate \(G\) rollouts using \(\pi_{\text{old}}\).
2. Compute rewards and normal GRPO response advantages.
3. Split responses into correct and incorrect groups.
4. If \(N^+=0\) or \(N^-=0\), skip NTHR for this prompt.
5. During the old-policy forward pass, save completion-token final hidden states.
6. Build \(V_x^*\), the unique generated-token vocabulary for this prompt.
7. Build prediction-error matrices \(G^+\) and \(G^-\).
8. Build positive hidden matrix \(H^+\).
9. Compute

\[
M^+=(G^+)^\top H^+.
\]

10. Compute positive token scores

\[
s^+ = \operatorname{rowsum}((G^+M^+)\odot H^+).
\]

11. Average \(s^+\) separately for each positive response to obtain \(\bar s_i^+\).
12. Compute

\[
\tau=\beta\min_i\bar s_i^+.
\]

13. Compute all negative token scores

\[
s^-=\operatorname{rowsum}((G^-M^+)\odot H^-).
\]

14. Form

\[
\text{mask}_{\text{NTHR}} = (s^- > \tau).
\]

15. Compute

\[
\eta=2|0.5-p|.
\]

16. Multiply the normal negative advantage by \(\eta\) only at masked negative-token positions.
17. Run the normal GRPO clipped policy loss with these token-level advantages.

---

## 12. PyTorch-style pseudocode

```python
def nthr_token_advantages(
    rewards,             # [G], binary 0/1
    response_adv,        # [G], standard GRPO response advantages
    token_ids,           # list[G] of [Ti]
    hidden,              # list[G] of [Ti, d], old-policy final hidden states
    probs_on_vstar,      # list[G] of [Ti, V*], old-policy probabilities
    token_vstar_index,   # list[G] of [Ti], sampled-token column in V*
    beta=1.0,
):
    G = len(rewards)

    pos_ids = [i for i in range(G) if rewards[i] == 1]
    neg_ids = [i for i in range(G) if rewards[i] == 0]

    # NTHR requires both positive and negative responses.
    if len(pos_ids) == 0 or len(neg_ids) == 0:
        return [
            torch.full(
                (len(token_ids[i]),),
                response_adv[i],
                device=hidden[i].device,
                dtype=hidden[i].dtype,
            )
            for i in range(G)
        ]

    p = len(pos_ids) / G
    eta = 2.0 * abs(0.5 - p)

    def prediction_error(prob, sampled_col):
        # prob: [T, V*]
        g = -prob.clone()
        rows = torch.arange(prob.shape[0], device=prob.device)
        g[rows, sampled_col] += 1.0
        return g

    G_pos_list = []
    H_pos_list = []
    pos_lengths = []

    for i in pos_ids:
        g_i = prediction_error(
            probs_on_vstar[i],
            token_vstar_index[i],
        )
        G_pos_list.append(g_i)
        H_pos_list.append(hidden[i])
        pos_lengths.append(g_i.shape[0])

    G_pos = torch.cat(G_pos_list, dim=0)   # [T+, V*]
    H_pos = torch.cat(H_pos_list, dim=0)   # [T+, d]

    # Positive aggregate matrix.
    M_pos = G_pos.T @ H_pos                # [V*, d]

    # Positive anchor scores.
    s_pos_token = ((G_pos @ M_pos) * H_pos).sum(dim=-1)

    anchor_scores = []
    offset = 0
    for T_i in pos_lengths:
        anchor_scores.append(
            s_pos_token[offset:offset + T_i].mean()
        )
        offset += T_i

    tau = beta * torch.stack(anchor_scores).min()

    # Start from standard response-level GRPO advantages.
    token_adv = [
        torch.full(
            (len(token_ids[i]),),
            response_adv[i],
            device=hidden[i].device,
            dtype=hidden[i].dtype,
        )
        for i in range(G)
    ]

    # Selectively weaken negative-token penalties.
    for j in neg_ids:
        G_neg_j = prediction_error(
            probs_on_vstar[j],
            token_vstar_index[j],
        )
        H_neg_j = hidden[j]

        s_neg_j = ((G_neg_j @ M_pos) * H_neg_j).sum(dim=-1)
        nthr_mask = s_neg_j > tau

        token_adv[j][nthr_mask] *= eta

    return token_adv
```

The returned `token_adv[i][k]` should replace the usual broadcasted response advantage in the GRPO loss.

---

## 13. Practical tensor shapes

For one prompt:

```text
G                         number of rollouts
T+                        total positive completion tokens
T-                        total negative completion tokens
d                         hidden dimension
V*                        unique generated-token vocabulary size

G_pos       [T+, V*]      positive prediction-error vectors
H_pos       [T+, d]       positive final-layer hidden states
M_pos       [V*, d]       positive aggregate matrix

G_neg       [T-, V*]      negative prediction-error vectors
H_neg       [T-, d]       negative hidden states
s_neg       [T-]          NTHR score of every negative token
```

The critical vectorized computation is only

```python
M_pos = G_pos.T @ H_pos
scores = ((G_tokens @ M_pos) * H_tokens).sum(-1)
```

---

## 14. Important implementation details

### Use completion tokens only

The sums in NTHR are over generated response tokens, not prompt tokens.

Mask padding and prompt positions out of:

- \(G^+\),
- \(G^-\),
- \(H^+\),
- \(H^-\),
- positive-response averages.

### Use the rollout / old-policy snapshot

Compute the NTHR scores and mask from the old-policy quantities associated with the sampled rollouts, then keep the selected-token mask fixed while optimizing that rollout batch.

This naturally fits the existing GRPO old-policy forward pass.

### Do not detach the final policy loss

The NTHR score itself is only used to construct a token mask / weight. It does not need backpropagation.

Conceptually:

```python
with torch.no_grad():
    nthr_mask = compute_nthr_mask(...)

loss = grpo_loss(policy_logits, token_adv_modified)
loss.backward()
```

### Numerical precision

The hidden-state inner products and matrix multiplication can accumulate over many tokens. It is safer to accumulate `M_pos` and scores in FP32 even if the model forward pass uses BF16.

For example:

```python
M_pos = G_pos.float().T @ H_pos.float()
```

### Empty/extreme groups

If

\[
p=0
\quad\text{or}\quad
p=1,
\]

there is no useful NTHR comparison.

Do not evaluate formulas containing

\[
\sqrt{p(1-p)}
\]

without an existing GRPO safeguard.

---

## 15. Minimal patch to an existing GRPO trainer

If your trainer already does this:

```python
advantages = response_adv[:, None].expand_as(completion_mask)
loss = grpo_loss(log_probs, old_log_probs, advantages)
```

NTHR changes it conceptually to:

```python
advantages = response_adv[:, None].expand_as(completion_mask).clone()

with torch.no_grad():
    nthr_mask, eta = compute_nthr_mask_and_eta(...)

# nthr_mask is true only for selected tokens from incorrect rollouts
advantages[nthr_mask] *= eta

loss = grpo_loss(log_probs, old_log_probs, advantages)
```

So the main engineering work is **not** changing GRPO itself.  
It is computing `nthr_mask` efficiently from old-policy hidden states and token prediction-error vectors.

---

## 16. Default method configuration

Use these values first if reproducing the main method:

\[
\boxed{\beta=1}
\]

and

\[
\boxed{
\eta=2|0.5-p|
}
\]

with

\[
p=N^+/G.
\]

The paper uses 8 rollouts per prompt in its main experiments, but NTHR itself is not mathematically restricted to \(G=8\).

---

## 17. Core algorithm in one block

\[
\boxed{
\begin{aligned}
g_{i,k}
&=
e_{y_{i,k}}
-
\pi_{\text{old}}(\cdot\mid x,y_{i,<k}),
\\[3pt]
M^+
&=
\sum_{i\in\mathcal P}
\sum_k
g^+_{i,k}(h^+_{i,k})^\top,
\\[3pt]
s_{i,k}
&=
g_{i,k}^\top M^+ h_{i,k},
\\[3pt]
\bar s_i^+
&=
\frac{1}{|y_i^+|}
\sum_k s^+_{i,k},
\\[3pt]
\tau
&=
\beta
\min_{i\in\mathcal P}
\bar s_i^+,
\\[3pt]
m^-_{j,k}
&=
\mathbbm 1[s^-_{j,k}>\tau],
\\[3pt]
\eta
&=
2|0.5-p|,
\\[3pt]
\hat A^{\text{NTHR}}_{j,k}
&=
\begin{cases}
\eta\hat A_j,&m^-_{j,k}=1,\\
\hat A_j,&m^-_{j,k}=0.
\end{cases}
\end{aligned}
}
\]

Then use \(\hat A^{\text{NTHR}}_{i,k}\) in the ordinary GRPO/PPO clipped policy objective.

That is the complete core of NTHR.
