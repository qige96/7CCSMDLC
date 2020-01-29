## A Decentralized Resource Allocation System

### Abstract

Resource allocation is a commonly seen task. Typical cases are module selection, or labor division and allocation. This proposal first describe one widely used resource allocation model implemented by centralized architecture, followed by its weakness, and then propose a decentralized improvement by blockchain technology.

### A resource allocation model

One process for resource allocation task is that, initially users claim their request, and then users could get the resources they needed if there are enough resource, or otherwise resources would be randomly allocated to users demanded. After that users could release the resources they don't want any more, and compete for spare resources, usually in a "First Come First Serve" order.

Often this model implemented by a client-server architecture. Inherently it has some weakness. First of all, allocation results might be manipulated by central authority. And from the technical perspective, Dos attack to the central server, or burst traffic, is always an annoying problem. 

### Improvement by blockchain

We could improve the model above by blockchain technology. Since blockchain is a decentralized architecture, the weakness inherited from C/S architecture would be solved naturally. 

To begin with, some terminologies need to be defined.

**Definition 1   Request** 

We define a  resource request  as a vector of [REQ, res_id, user_id]. 

**Definition 2   Release** 

We define a resource release as a vector of [REL, res_id, user_id].

**Definition 3   Quote List** 

We define a resource quote list as a vector of [quote_of_res1, quote_of_res2, ...]

**Definition 4   Block**

A blockchain is a linked list of blocks that each node stores one copy. In this resource allocation system, the structure of block is defined as

| Feature       | Content                                       |
| ------------- | --------------------------------------------- |
| previous hash | hash of previous block                        |
| timestamp     | timestamp of block creation                   |
| quote list    | list that record the quote of all resources   |
| hash          | merkle hash of all transactions in this block |
| transactions  | a collection of resources request and release |

At the first stage, users claim and broadcast their requests. After deadline of the first stage, the system collect all the requests and allocate resources according to the resource quote list: if enough then allocate the resources to the user who needed, or otherwise randomly. This first stage don't adopt the "Fist Come First Serve" principle because that might result in severe competition among all users and thus network congestion. the result of the first stage become the genesis block.

At the second stage, users can release resource they don't want any more, by submit a release transaction. Also they can submit a request transaction, in this stage "Fist Come First Serve" principle is adopted. Each time a node mines a block, it will organize the transactions (both release and request) in a sequential order, and does addition and subtraction on resources quotes. Valid transactions will be accepted into blocks, while invalid transactions will be abandoned.

### Development Schedule

We use Ethereum as the infrastructure to develop this allocation system. Ethereum has encapsulated complicated procedures like encryption and decryption, and proof of work (stake), so  that we can focus on application layer. Also, we need a user friendly interface. Web app usually serves this purpose.

2 or 3 persons will do the blockchain development, and 2 persons do the user interface will be enough. The whole project could be completed in less than 1 month, since this is just a prototype demo.

| Time     | Job                   |
| -------- | --------------------- |
| 3rd Feb  | Submit proposal       |
|          |                       |
| 18th Mar | Engineering Completed |



