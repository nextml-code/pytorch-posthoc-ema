# Post-hoc EMA

It is well known that exponential moving average (EMA) of
model weights plays an important role in generative image
synthesis [55, 78], and that the choice of its decay parameter
has a significant impact on results [32, 55].
Despite its known importance, little is known about the
relationships between the decay parameter and other aspects
of training and sampling. To analyze these questions, we
develop a method for choosing the EMA profile post hoc, i.e.,
without the need to specify it before the training. This allows
us to sample the length of EMA densely and plot its effect
on quality, revealing interesting interactions with network
architecture, training time, and classifier-free guidance.
Further details, derivations, and discussion on the equations and methods in this section are included in Appendix C.

3.1. Power function EMA profile
Traditional EMA maintains a running weighted average ˆθβ
of the network parameters alongside the parameters θ that are
being trained. At each training step, the average is updated
by ˆθβ(t) = β ˆθβ(t−1) + (1−β) θ(t), where t indicates the
current training step, yielding an exponential decay profile in
the contributions of earlier training steps. The rate of decay
is determined by the constant β that is typically close to one.
For two reasons, we propose using a slightly altered averaging profile based on power functions instead of exponential
decay. First, our architectural modifications tend to favor
longer averages; yet, very long exponential EMA puts nonnegligible weight on initial stages of training where network
parameters are mostly random. Second, we have observed
a clear trend that longer training runs benefit from longer
EMA decay, and thus the averaging profile should ideally
scale automatically with training time.
Both of the above requirements are fulfilled by power
functions. We define the averaged parameters at time t as

θγ(t) =
R t
0
τ
γ
θ(τ ) dτ
R t
0
τ
γ dτ
=
γ + 1
t
γ+1 Z t
0
τ
γ
θ(τ ) dτ

where the constant γ controls the sharpness of the profile.
With this formulation, the weight of θt=0 is always zero.

This is desirable, as the random initialization should have no
effect in the average. The resulting averaging profile is also
scale-independent: doubling the training time automatically
stretches the profile by the same factor.
To compute ˆθγ(t) in practice, we perform an incremental
update after each training step as follows:
ˆθγ(t) = βγ(t)
ˆθγ(t − 1) +
1 − βγ(t)
θ(t)
where βγ(t) = (1 − 1/t)
γ+1
.
(2)
The update is thus similar to traditional EMA, but with the
exception that β depends on the current training time.2
Finally, while parameter γ is mathematically straightforward, it has a somewhat unintuitive effect on the shape
of the averaging profile. Therefore, we prefer to parameterize the profile via its relative standard deviation
σrel, i.e., the “width” of its peak relative to training time:
σrel = (γ + 1)1/2
(γ + 2)−1
(γ + 3)−1/2
. Thus, when reporting, say, EMA length of 10%, we refer to a profile with
σrel = 0.10 (equiv. γ ≈ 6.94)

## Synthesizing novel EMA profiles after training

Our goal is to allow choosing γ, or equivalently σrel, freely
after training. To achieve this, we maintain two averaged
parameter vectors ˆθγ1
and ˆθγ2
during training, with constants
γ1 = 16.97 and γ2 = 6.94, corresponding to σrel of 0.05 and
0.10, respectively. These averaged parameter vectors are
stored periodically in snapshots saved during the training
run. In all our experiments, we store a snapshot once every
∼8 million training images, i.e., once every 4096 training
steps with batch size of 2048.
To reconstruct an approximate ˆθ corresponding to an arbitrary EMA profile at any point during or after training,
we find the least-squares optimal fit between the EMA profiles of the stored ˆθγi
and the desired EMA profile, and take
the corresponding linear combination of the stored ˆθγi.

We note that post-hoc EMA reconstruction is not limited
to power function averaging profiles, or to using the same
types of profiles for snapshots and the reconstruction. Furthermore, it can be done even from a single stored ˆθ per
snapshot, albeit with much lower accuracy than with two
stored ˆθ. This opens the possibility of revisiting previous
training runs that were not run with post-hoc EMA in mind,
and experimenting with novel averaging profiles, as long as
a sufficient number of training snapshots are available.

## Analysis

Armed with the post-hoc EMA technique, we now analyze
the effect of different EMA lengths in various setups

Figure 5a shows how FID varies based on EMA length
in configurations B–G of Table 1. We can see that the optimal EMA length differs considerably between the configs.
Moreover, the optimum becomes narrower as we approach
the final config G, which might initially seem alarming.
However, as illustrated in Figure 5b, the narrowness of
the optimum seems to be explained by the model becoming
more uniform in terms of which EMA length is “preferred”
by each weight tensor. In this test, we first select a subset
of weight tensors from different parts of the network. Then,
separately for each chosen tensor, we perform a sweep where
only the chosen tensor’s EMA is changed, while all others
remain at the global optimum. The results, shown as one
line per tensor, reveal surprisingly large effects on FID. Interestingly, while it seems obvious that one weight tensor
being out-of-sync with the others can be harmful, we observe
that in CONFIG B, FID can improve as much as 10%, from
7.24 to ∼6.5. In one instance, this is achieved using a very
short per-tensor EMA, and in another, a very long one. We
hypothesize that these different preferences mean that any
global choice is an uneasy compromise. For our final CONFIG G, this effect disappears and the optimum is sharper: no
significant improvement in FID can be seen, and the tensors
now agree about the optimal EMA. While post-hoc EMA
allows choosing the EMA length on a per-tensor basis, we
have not explored this opportunity outside this experiment.
Finally, Figure 5c illustrates the evolution of the optimal
EMA length over the course of training. Even though our
definition of EMA length is already relative to the length of
training, we observe that the optimum slowly shifts towards
relatively longer EMA as the training progresses.

Figure 4. Top: To simulate EMA with arbitrary length after training, we store a number of averaged network parameter snapshots
during training. Each shaded area corresponds to a weighted average of network parameters. Here, two averages with different power
function EMA profiles (Section 3.1) are maintained during training
and stored at 8 snapshots. Bottom: The dashed line shows an example post-hoc EMA to be synthesized, and the purple area shows the
least-squares optimal approximation based on the stored snapshots.
With two averaged parameter vectors stored per snapshot, the mean
squared error of the reconstructed weighting profile decreases extremely rapidly as the number of snapshots n increases, experimentally in the order of O(1/n4
). In practice, a few dozen snapshots
is more than sufficient for a virtually perfect EMA reconstruction

Figure 5. (a) FID vs. EMA length for our training configs on ImageNet-512. CONFIG A uses traditional EMA, and thus only a single point
is shown. The shaded regions indicate the min/max FID over 3 evaluations. (b) The orange CONFIG B is fairly insensitive to the exact EMA
length (x-axis) because the network’s weight tensors disagree about the optimal EMA length. We elucidate this by letting the EMA length
vary for one tensor at a time (faint lines), while using the globally optimal EMA length of 9% for the others. This has a strong effect on FID
and, remarkably, sometimes improves it. In the green CONFIG G, the situation is different; per-tensor sweeping has a much smaller effect,
and deviating from the common optimum of 13% is detrimental. (c) Evolution of the EMA curve for CONFIG G over the course of training
