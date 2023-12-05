---
layout: post
title:  "CUDA vs ROCm: A Case Study"
date: 2023-12-05 17:01:00
tags: HPC GPGPU C++
---

How far along is AMD's ROCm in catching up to Cuda? AMD has been on this race for a while now, with ROCm debuting 7 years ago. Answering this question is a bit tricky though. CUDA isn’t a single piece of software—it’s an entire ecosystem spanning compilers, libraries, tools, documentation, Stack Overflow/forum answers, etc. Today, I’m going to zoom in on a particular slice of these vast ecosystems, that hopefully sheds some light on the current state-of-affairs of the broader landscape. That component today will be the random number generation libraries: cuRAND and rocRAND, part of the suite of around ten libraries that come standard on both systems.

I want to preface this by saying: this is going to be an opinionated article. Though in my defense, I didn't start from a conclusion. I originally approached this as part of my graduate research into performance-portable softwares, in usual dry, academic way. But as I worked, I realized I was forming opinions that I can not really express through an academic paper. Hence, this post.

One of the key advantages of rocRAND is it is open-source. So let's start at their GitHub repo first.

## Design
Going through the README, one of the first things you notice is AMD actually offers two random number libraries: rocRAND and hipRAND, the latter being a thin client that chooses cuRAND or rocRAND depending on the platform. So, for today's discussion, we'll set aside hipRAND.

In the Requirements section, for AMD platforms, you'll find ROCm listed, which is expected. However, clicking on the ROCm link leads to a 404 error page.For CPU runs, you need something referred to as "HIP-CPU". This link thankfully works, and it seems to be- "An implementation of HIP that works on CPUs, across OSes."

Let's pause for a moment. We're not even halfway through the README and we have already seen 3 different platforms from AMD- ROCm, HIP, HIP-CPU. I really wonder about the necessity or the wisdom behind this fragmentation- splitting HIP in particular. A single standard or library like SYCL or Kokkos seems to support multiple hardware platforms just fine under one codebase. To me this felt like a half-hearted attempt to tick one more box in a head-to-head battle with (intel-supported) SYCL. And I say half-hearted because HIP-CPU has been under development for more than 3 years, and this is the first paragraph of its README: "Please note the library is being actively developed, and is known to be incomplet; it might also be incorrekt and there could be a few bad bugs lurking." Let's return to our focus on rocRAND.

One of the key challenges in developing a parallel, reproducible random number library is ensuring statistical robustness. This might not matter for most users, but for applications like Brownian simulations, a weak generator can silently wreak havoc. Rigorous testing with standard, widely accepted statistical frameworks is crucial - something cuRAND of course does. However, I couldn't find any extensive discussion on this for rocRAND, aside from two self-written tests. There's mention of a statistical test suite in the README, but again, that link leads to a 404 error.

It's not looking great, but at this point, I found a feature that cuRAND doesn't have, a Python API! It's an interesting choice: to attach such a high-level language interface for such a low-level library. So let's go to the documentation and see what's it for, shall we?

## Documentation

{% include figure.html path="assets/img/rocm/py.png" class="img-fluid rounded z-depth-1 mx-auto d-block" caption="Figure 1: rocRAND's Python API."%}

That's it! That's the entirety of the Python API documentation – and no, those headers aren’t clickable. [This is it](https://rocm.docs.amd.com/projects/rocRAND/en/latest/python_api.html)!!!

So, that was a bonus feature. What about the C++ API documentation? well, it exists, but it's hardly any different. It's almost entirely just a dump of function docstrings, with same comment copy/pasted for all the functions. And this mindless copy/pasting has predictable result- you'll find, for example, the "documentation" mention 64 bit int return type for a function while it actually returns 32-bit. 

Frankly, this isn't just bad documentation; this is horrendous. There is no attempt anywhere to introduce or explain anything: just data dumps and lists. You get the sense, once again, that this "documentation work" was another box for someone to tick, without any consideration paid to a potential user of the software. 

But the code follows the same API as cuRAND. So someone familar with cuRAND will be able to manage. (I guess that was the assumption of documentation authors, they didn't imagine anyone coming straight to AMD ecosystem.) Let's look at that code next.

## Performance

I'll start with a real-world benchmark,  using a classic example of GPGPU programming: Ray tracing in one weekend in cuda ([Github](https://github.com/rogerallen/raytracinginoneweekendincuda)). For meaningful performance comparison of random number libraries, we need a program that uses random numbers beyond just the initialization phase. Ray tracer is a good example of that. Both libraries offer a variety of generators; for this test, I chose Philox.

{% include figure.html path="assets/img/rocm/comb.jpg" class="img-fluid rounded z-depth-1 mx-auto d-block" caption="Figure 2:  Time taken to render the image on the right by cuRAND and rocRAND libraries (left)"%}

4.03 seconds vs 5.5s- the raytracer with the rocRAND version is 37% slower. Remember this isn't a micro-benchmark of just random number generation part, the timings are for whole program. I think this is a pretty substantial slowdown.

The benchmark was performed on an Nvidia V100 GPU. Is that fair? I think yes, especially since rocRAND’s developers [claimed](https://streamhpc.com/blog/2017-11-29/learn-amds-prng-library-developed-rocRAND/) to have performance parity with cuRAND on Nvidia GPUs. But maybe cuRAND has some hardware-specific optimizations? I really don't think that's the case. Philox algorithm isn't that complicated, it doesn't really need any advanced GPU primitives. But don't take just my word for it: our lab made a pretty simple implementation of Philox, (you can find it here [here](https://github.com/msu-sparta/OpenRAND/blob/main/include/openrand/phillox.h)), it is orders of magnitude smaller than rocRAND's implementation in terms of LOC, yet it performs on par with CuRAND (4.09 seconds).


Still, it's just one benchmark. I'm sure there are other hardware-software combinations where this performance gap disappears. But, just to ensure that the ray tracer isn't some outlier, I wrote a pretty basic 2D brownian dynamics simulation code. The story is even worse here for rocRAND, 6.30 seconds vs cuRAND's 4.23- a 48% slowdown.

## Final Thoughts

A thought occurred to me recently, you can almost put a dollar amount to the value of cuda now. It's a bit silly, but humour me: let's take the value of AI riches that went almost exclusively to Nvidia since the ChatGPT shockwave (say around 700B USD), and let's assume Cuda is responsible for x% of it. Here's the thing: no matter how low you think that x is (within reason), it's going to be an astronomical sum! I'm sure this math didn't escape AMD.

Many assumed this will be a big wake-up call for AMD, their [Carthage must be destroyed](https://www.vanityfair.com/news/2016/06/how-mark-zuckerberg-led-facebooks-war-to-crush-google-plus) moment that radically alters their well-known laid-back attitude to software. There are hints of this shift in their recent events and press releases, and I hope this trend continues.

But in my little corner of GPGPU world, I'm yet to see any meaningful movement in that regard. As I wrote this article, I took a cursory glance at Intel's [documentation](https://spec.oneapi.io/versions/1.2-rev-1/elements/oneMKL/source/domains/rng/onemkl-rng-overview.html) for SYCL (a competitor of HIP) on this topic- a clean, well-organized, professional site- as you'd expect. 

Like many, I’m looking forward to a real showdown in the GPGPU space someday- I'm just not sure that will necessarily be between Nvidia and AMD. 



<!-- Judging by the recent stock market movements since chatgpt shockwave, I’m guessing a lot of people outside the niche GPGPU community is interested (invested) in thsi question. , as this has been one of the key differentiators that put the recent AI riches almost exclusively yo the coffer of Nvidia, while AMD watched from sideline. -->

