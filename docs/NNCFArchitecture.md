# NNCF Architectural Overview

## Introduction

Neural Networks Compression Framework is a set of compression algorithms and tools to implement compression algorithms that is designed to work atop PyTorch.
In essence, all of the compression algorithms present in NNCF do certain manipulations with the data inside the control flow graph of a DNN - be it the process of quantizing the values of an input tensor for a fully connected layer, or setting certain values of a convolutional layer to zero, etc.
A general way to express these manipulations is by using hooks inserted in specific points of the DNN control flow graph.

## NNCFGraph

To abstract away the compression logic from specifics of the backend, NNCF builds an `NNCFGraph` object for each incoming model object to be compressed.
`NNCFGraph` is a wrapper over a regular directed acyclic graph that represents a control flow/execution graph of a DNN.
Each node corresponds to a call of a backend-specific function ("operator").
It is built both for the original, unmodified model, and for the model with compression algorithms applied (which, in general, may have additional operations when compared to the original model).
