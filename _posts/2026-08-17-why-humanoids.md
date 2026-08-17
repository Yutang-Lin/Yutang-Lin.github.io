---
layout: post
title: Why Humanoids?
date: 2026-08-17 00:00:00-0400
description: Whole-body physicality, not the humanoid form factor, is what makes humanoids fundamentally different — and why I think we do not yet have a general-purpose whole-body model.
tags: Robotics
categories: Robotics
related_posts: false
---

*Why humanoids?*

A common answer is that humanoids are universal platforms because our environments are designed for humans. Functionally, yes. To me, this probably answers *"can humanoids?"*, rather than *"why humanoids?"* It establishes feasibility, but not uniqueness.

I think the more fundamental answer is *whole-body physicality*.

Think about opening a heavy door. We do not simply pull the handle with an arm. We reposition our feet, lean with the body, and transmit force through the entire kinematic chain. In other tasks, we may deliberately create contacts with the shoulder, or torso etc. *The body itself is part of the manipulation system.* By whole-body physicality, I mean that the entire body becomes part of the physical solution to a task. A humanoid does not merely manipulate with its hands while using its legs as a mobile base.

Many humanoid models are called whole-body because they generate or track motions over the entire body. But controlling more degrees of freedom is not, by itself, whole-body problem solving. If a high-level model specifies a whole-body reference and a low-level controller tracks it, then conceptually this is not very different from a mobile-base bimanual robot tracking a base trajectory and two arm trajectories.

The embodiment is different. The control problem is harder. But the role of the body in solving the task has not fundamentally changed. Whole-body tracking is not whole-body intelligence.

Task-specific reinforcement learning already gives us glimpses of the stronger notion. A policy can learn to lean, brace, redistribute weight, exploit momentum, or create unexpected contacts because these behaviors help accomplish the task. In such cases, the body is not merely following a prescribed motion. It is being used as a physical resource.

What I do not think we have today is a *general-purpose model that treats the whole body in this way*. A humanoid risks becoming little more than a very complicated mobile manipulator: the legs move the platform, the hands manipulate, and the rest of the body is something to keep stable or track accurately.

That is not the reason I find humanoids interesting. The real promise of humanoids is that *the whole body becomes part of the solution space*. And by that definition, I do not think we have a general-purpose whole-body model yet.
