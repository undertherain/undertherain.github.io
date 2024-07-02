---
title: "Doing things with static word embeddings"
date: 2023-03-15
description: ""
showDate: false
---

Once upon a time static word embeddings were a pretty hot area in NLP research.
Well, they didn't go anywhere actually, we still have an embedding layer at the input of most modern LLMs.
But now it's "just there" and most of research focus is on other things.
Back then though, embeddings were trained in a simpler way - just predicting if two words will appear close to each other
based on cosine distance between "input" and "output" set of embeddings.
And then one would take only the embeddings table (say the input one), slap an LSTM on top of it and train for sentiment analysis or whatever downstream task you fancy.
It was crude, but worked well enough!

We did out share of related work as well - building better datasets, making a toolkit for automating evaluation of embeddings,
training embeddings using subword and subcharacter (!) information.
I hope one day I'll write a nice all-explaining blog-post - now everything is scattered across a number of papers.
The good news are, you can go to https://vecto.space/ and find all papers listed, all codes and datasets - downloadable and some work having nice easy-to-read writeups. Cheers!
