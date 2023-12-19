---
layout: post
title:  "CUDA vs ROCm: A Case Study"
date: 2023-12-05 17:01:00
tags: HPC GPGPU C++
---

How far along is AMD's ROCm in catching up to Cuda? AMD has been on this race for a while now, with ROCm debuting 7 years ago. Answering this question is a bit tricky though. CUDA isn’t a single piece of software—it’s an entire ecosystem spanning compilers, libraries, tools, documentation, Stack Overflow/forum answers, etc. Today, I’m going to zoom in on a particular slice of these vast ecosystems, the random number generation libraries: cuRAND and rocRAND, part of the suite of around ten libraries that come standard on both systems. Hopefully, this sheds some light on the current state-of-affairs of the broader landscape. 

Most of these observations grew out of my work on a research project a few months ago. As I worked, I realized I was forming some pretty strong takes that I can't really put in an academic paper. So here I am.

One of the key advantages of rocRAND is it is open-source. So let's start at their [GitHub repo](https://github.com/ROCm/rocRAND) first.

## Design
Going through the README, one of the first things you notice is AMD actually offers two random number libraries: rocRAND and hipRAND, the latter being a thin client that chooses cuRAND or rocRAND depending on the platform. So, for today's discussion, we'll set aside hipRAND.

Next comes a list of random number generators implemented in the library. You won't find a discussion about them here (or anywhere else for that matter), Just a list of names. Moving on, in the Requirements section, ROCm is listed as a dependency for AMD platforms, as expected. However, clicking on the ROCm link leads to the first 404 error on this page. To run this library on CPU, you need something referred to as "HIP-CPU". This link thankfully works, and the tagline of its Github repo reads- "An implementation of HIP that works on CPUs, across OSes."

Let's pause for a moment. We're not even halfway through the README and we have already seen 3 different platforms from AMD- ROCm, HIP, HIP-CPU. I really wonder about the necessity or the wisdom behind this fragmentation- splitting HIP in particular. A single standard or library like SYCL or Kokkos seems to support multiple hardware platforms just fine under one codebase. To me this felt like a half-hearted attempt to tick one more box in a head-to-head battle with (intel-supported) SYCL. And I say half-hearted because HIP-CPU has been under development for more than 3 years, last commit pushed 3 months ago, and this is the first paragraph of its README: "Please note the library is being actively developed, and is known to be incomplet; it might also be incorrekt and there could be a few bad bugs lurking." Let's return to our focus on rocRAND.

One of the key challenges in developing a parallel, reproducible random number library is ensuring statistical robustness. This might not matter for most users, but for applications like Brownian simulations, a weak generator can silently wreak havoc. Rigorous testing with standard, widely accepted statistical frameworks is crucial - something cuRAND of course does. However, I couldn't find any discussion on this for rocRAND, aside from two self-written simple tests. There's mention of a statistical test suite in the README, but again, that link leads to a 404 error.

It's not looking great, but at this point, I found a feature that cuRAND doesn't have, a Python API! It's an interesting choice: to attach such a high-level language interface for such a low-level library. So let's go to the documentation and see what's it for, shall we?

## Documentation

{% include figure.html path="assets/img/rocm/py.png" class="img-fluid rounded z-depth-1 mx-auto d-block" caption="Figure 1: rocRAND's Python API."%}

That's it! That's the entirety of the Python API documentation – and no, those headers aren’t clickable. [This is it](https://rocm.docs.amd.com/projects/rocRAND/en/latest/python_api.html)!

So, that was a bonus feature. What about the C++ API documentation? well, it exists, but it's hardly any different. The API reference is almost entirely just a dump of function docstrings, with same comment copy/pasted for all the functions. And this mindless copy/pasting has predictable result- you'll find, for example, the "documentation" mention 64 bit int return type for a function while it actually returns 32-bit. 

The Programming Guide again starts (and ends) with the list of generators, with only one piece of extra information here, whether a generator is for pseudo-random or quasi-random number generation. The next (and final) section is titled "Ordering", and the very first sentence starts talking about "how results are ordered in global memory." If you just thought- wait, what results? that's a very valid response. You *might* eventually figure out they are talking about the host-side API that generates a buffer of random numbers on device. Being GPU, it uses multiple threads behind the scene, and ordering here refers to how to order the numbers coming out of each thread in the output buffer. They list 5 ways of doing it, after commenting how this choice impacts performance and reproducibility. Go on, [read about them a little bit](https://rocm.docs.amd.com/projects/rocRAND/en/latest/programmers_guide.html#), you'll soon discover a pretty interesting relationship between them. For the lazy among you, here's a clue:  

{% include figure.html path="assets/img/rocm/Spiderman.png" class="img-fluid rounded z-depth-1 mx-auto d-block" caption="" style="max-width: 50%;" %}

They are all the same! Of course, they don't say that directly, it's another little thing for you to figure out. (well, technically I can't say "all" are same, becuase they don't mention the fifth one anywhere else in the page.)

Frankly, this isn't just bad documentation; this is horrendous. There is no attempt anywhere to introduce or explain anything: just data dumps and lists. You get the sense, once again, that this "documentation work" was another box for someone to tick, without any consideration paid to a potential user of the software. 

But the code follows the same API as cuRAND. So someone familar with cuRAND will be able to manage eventually. Let's look at how that code fares against cuRAND next.

## Performance

I'll start with a real-world benchmark,  using a classic example of GPGPU programming: Ray tracing in one weekend in cuda ([Github](https://github.com/rogerallen/raytracinginoneweekendincuda)). For meaningful performance comparison of random number libraries, we need a program that uses random numbers beyond just the initialization phase. Ray tracer is a good example of that. Both libraries offer a variety of generators; for this test, I chose Philox.

{% include figure.html path="assets/img/rocm/comb.jpg" class="img-fluid rounded z-depth-1 mx-auto d-block" caption="Figure 2:  Time taken to render the image on the right by cuRAND and rocRAND libraries (left)"%}

4.03 seconds vs 5.5s- the raytracer with the rocRAND version is 37% slower. Remember this isn't a micro-benchmark of just random number generation part, the timings are for whole program. With that in mind, I think this is a pretty substantial slowdown.

The benchmark was performed on an Nvidia V100 GPU. Is that fair? I think yes, especially since rocRAND’s developers [claimed](https://streamhpc.com/blog/2017-11-29/learn-amds-prng-library-developed-rocRAND/) to have performance parity with cuRAND on Nvidia GPUs. But maybe cuRAND has some hardware-specific optimizations? I really don't think that's the case. Philox algorithm isn't that complicated, it doesn't really need any advanced GPU primitives. But don't take just my word for it: our lab made a pretty simple implementation of Philox, (you can find it [here](https://github.com/msu-sparta/OpenRAND/blob/main/include/openrand/philox.h)), it is orders of magnitude smaller than rocRAND's implementation in terms of LOC, yet it performs on par with CuRAND (4.09 seconds).


Still, it's just one benchmark. I'm sure there are other hardware-software combinations where this performance gap disappears. But, just to ensure that the ray tracer isn't some outlier, I wrote a pretty basic 2D brownian dynamics simulation code. The story is even worse here for rocRAND, 6.30 seconds vs cuRAND's 4.23- a 48% slowdown.

## Final Thoughts

After the ChatGPT phenomenon, there has recently been lots of focus on Nvidia's "CUDA moat". As we all watched the vast AI riches going almost exclusively to Nvidia thanks mostly to that moat, many assumed this will be a big wake-up call for AMD, their [Carthage must be destroyed](https://www.vanityfair.com/news/2016/06/how-mark-zuckerberg-led-facebooks-war-to-crush-google-plus) moment that radically alters their well-known laid-back attitude to software. There are hints of this shift in their recent events and press releases, and I hope this trend continues.

But in my little corner of HPC world, I’m yet to see any meaningful movement in that regard. And AMD needs to hurry up- as I wrote this article, I took a cursory glance at Intel’s [documentation](https://spec.oneapi.io/versions/1.2-rev-1/elements/oneMKL/source/domains/rng/onemkl-rng-overview.html)  for SYCL (a competitor of HIP) on this topic- a clean, well-organized, professional site- as you’d expect.

Like many, I’m looking forward to a real showdown in the GPGPU space someday- I'm just not sure that will necessarily be between Nvidia and AMD. 



<!-- Judging by the recent stock market movements since chatgpt shockwave, I’m guessing a lot of people outside the niche GPGPU community is interested (invested) in thsi question. , as this has been one of the key differentiators that put the recent AI riches almost exclusively yo the coffer of Nvidia, while AMD watched from sideline. -->

