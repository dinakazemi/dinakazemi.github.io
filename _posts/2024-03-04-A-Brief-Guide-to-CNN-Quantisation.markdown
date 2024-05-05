---
layout: post
title:  "A Brief Guide to CNN Quantisation"
date:   2024-03-04 17:22:15 +1100
categories: CNN Quantisation
---
When I first started learning about quantisation of CNNs, I found myself against a dozen papers out there that went into the details of their design and implementation. This made me even more confused; I needed a more higher level view first, a step back to figure out **what** quantisation was and **how** it worked before being drenched into all the math and code. This blog post is here to, hopefully, provide that starting point for anyone who is interested.

---
Given the ever growing computational and memory requirements of Covolutional Neural Networks (CNNs), they are not suitable for deployment on resource-constrained devices such as mobile phones. As such, "quantisation" has been used as one of the state-of-the-art methods to decrease their energy and memory consumption. In this blog post , I’ll share insights from the [paper](https://arxiv.org/abs/1808.04752) "A Survey on Methods and Theories of Quantized Neural Networks" by Yunhui Guo, which provides a thorough exploration of the different facets of quantized neural networks (QNNs). This includes an in-depth look at common quantization methods, challenges of QNNs, and future trends.
> **_NOTE:_**  This article assumes you have some basic understanding of CNNs and their structure. I created a summary of a Kaggle tutorial on the topic called `computer-vision-basics` [here](https://github.com/dinakazemi/CNN-tutorials).

Before discussing the deatils of the paper, let us address the elephant in the room 🐘:
## What is Quantisation?
Quantisation is one of the more popular methods in optimising CNNs for resource-constrained hardware. It involves reducing the precision of model paramaters such as weights and activations, instead of using the full 32-bit floating point values. Switching to low-bitwidth values such as 8-bit, 4-bit, or even binary integers allows the model to use bitwise operations to perform the forward and backpropogation, resulting in an increase in efficiency. Moreover, quantised models benefit from reduced energy consumption, which is a critical consideration for battery-powered devices.

---
The paper classifies the different quantisation techniques into two broad categoreis: **deterministic** and **stochastic**. Deterministic methods create a one-to-one mapping of full precision values to their quantised versions using pre-defined functions or rules, while stochastic techniques introduce randomness in the quantisation process by selecting quantised values based on probability distributions of real values or by rounding them to their nearest discrete representations.  

So, being the more predictive out of the two, deterministic is better suited for implementation on dedicated hardware where we want minimum randomness introduced in the quantisation process. Stochastic on the other hand, can give us better insights into the distribution of the values and is, therefore, more useful for experimentation purposes, and/or perhaps in situations where introducing some randomness into the training process can improve network performance.  

Additionally, the paper details fixed codebook quantization where weights are quantized into predefined values, and adaptive codebook quantization where the codebook is learned from the data. Adaptive quantization is more flexible and can avoid ad hoc modifications to the training algorithm but may require more bits to represent the final codebook.

---
The fact that quantisation works and that we can get accuracies comparable to full precision CNNs really suggests that there is a lot of redundancy in CNN parameters and connections, implying that we can prune and quantize these networks without severe consequences on performance. Although better accuracy is always desirable, it should go hand-in-hand with accessibility; it's not always possible to use state-of-the-art GPUs and computational resources to get the job done.  

Since 2018 that this paper came out, there have been significant improvements in the field, with quantisation methods providing more acceleration not just for CNNs but also for LLMs and other types of artificial neural networks. There have also been studies to combine quantisation with other prominent methods such as Knowledge Distillation to improve their accuracy such this [paper](https://arxiv.org/abs/1802.05668).  

Despite this, it is surprising to me that there is no universal system for measuring and comparing the accuracy of QNNs among themselves or their full precision counterparts. Authors choose what datasets, network archtectures, and hardware to use to implement and test the accuracy of their design. So, I think a big step forward in the field would be to develop a more consistent measure of accuracy for QNNs, such as this recent [study](https://arxiv.org/pdf/2301.06193.pdf). 