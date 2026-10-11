---
layout: post
title: Where Is RLVR for Robotics Foundation Models?
date: 2026-10-10 00:00:00+0800
description: Is the bottleneck toward a “GPT moment” in robotics whether physical intelligence can become cheap to sample and reliably verifiable?
tags: Robotics Reinforcement-Learning
categories: Robotics
related_posts: false
---

Many frontier companies and institutions have released their own impressively scaled robotics foundation models. Several keywords are almost always popular: learning from human data, in-context learning, cross-embodiment transfer, million-hour-scale pre-training, and eventually, a scaling law for embodied intelligence.

While this seems like a replicated path toward the “GPT moment,” the fundamental difference between language modeling and physical intelligence should not be neglected.

Rewind back to around 2022, when academia and industry started to explore general-purpose instruction-following language models. It is easy to find similarities between that stage of language intelligence and today’s VLAs. Language models were first pre-trained on massive amounts of unlabeled Internet text, analogous to today’s attempts to learn from large-scale human video, and then fine-tuned on instruction-following data, somewhat analogous to teleoperation data in robotics. With sufficient task-specific supervision, language models became capable of completing increasingly diverse instructions. Similarly, today’s VLAs can already perform impressive laboratory tasks, such as stacking boxes, folding clothes, and making coffee, as long as sufficiently relevant robot data is available.

The analogy is obviously imperfect. Internet text already lives in the native output space of a language model, while human video does not directly provide robot actions, and even robot actions are not naturally shared across embodiments. Still, at a high level, the developmental trajectory looks surprisingly similar.

However, I personally think that one of the most important turning points toward the capabilities of today’s language models came from post-training with reinforcement learning, especially RLHF and, more recently, reinforcement learning with verifiable rewards. Pre-training gives the model a powerful prior; reinforcement learning gives it a mechanism to search, receive feedback, and improve beyond simply imitating the demonstrations in its dataset.

This is where a fundamental difference between language intelligence and physical intelligence starts to emerge.

Scalable reinforcement learning relies on several properties that language, code, and other digital domains happen to provide unusually well:

- **Broad task coverage under a relatively unified interface**
- **Highly efficient and massively parallel sampling of trajectories**
- **Cheap and reliable verification of outcomes**

Migrating the same recipe to robotics immediately runs into all three problems.

First, physical tasks are not naturally expressed under a single unified interface. Different embodiments have different action spaces, dynamics, sensing modalities, control frequencies, and physical constraints. More importantly, the space of meaningful physical tasks itself is difficult to enumerate or sample from. In language or code, generating another problem is cheap. In robotics, generating another meaningful physical situation often means physically constructing one.

Second, sampling an MDP in the physical world is expensive. A language model can generate thousands of candidate trajectories in parallel. A robot must actually execute the behavior, wait for the environment to evolve, reset objects, recover from failures, and sometimes involve human intervention. The gap in sampling throughput is enormous.

Third, and probably most fundamentally, physical trajectories are difficult to verify. In mathematics or code, correctness can often be reduced to an exact answer, a compiler, or a unit test. In robotics, task completion is rarely enough. A robot may successfully open a door while using an unstable, unsafe, or completely unnatural strategy. The real objective depends not only on the final state, but on the entire trajectory: stability, contact quality, robustness, recoverability, safety, and future controllability.

This makes me wonder whether the real bottleneck toward a “GPT moment” in robotics is not simply data or model scale, but whether we can construct an environment in which physical intelligence itself becomes cheap to sample and reliably verifiable.
