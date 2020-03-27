Hello everyone, this is Ricky Zhu, from BlockDance team. In this coursework, we chose to Implement a simple smart contract application on top of Ethereum Testnet, and the system we developed is called a Simple Decentralized Resource Allocation System. Now I am going to demonstrate our system prototype to you. 

To begin with, we need to set up a local Ethereum environment, because operations on global Ethereum TestNet are really slow, not suitable for demonstration purpose. Here I use Ganache as our local testing environment. Once set up, we are provided 10 addresses, each of which already has 100 Ethers. 

First of all, administrators should do some configurations. We assume that these three addresses are allowed to participate in this allocation event, so we need to put them into the setting file. Also, we need to specify how many resources are available. Here, we assume that resource 0 has 1 quote, resource 1 has 1 quote, resource 2 has 2 quote, etc. Lastly, we set that each user could at most hold 2 resources. Once all configuration done, we can deploy our contracts to Ethereum network.

Administrator send specific accounts to every user, and notifying all users that the allocation process starts. To interact with smart contracts living on blockchian, we need a user interface, so we develop a web application. But before we can use the web app, we still need to do something. This web app depends on an Ethereum wallet management software called metamask. It could be installed as a browser extension in Chrome, Firefox or other modern browsers. Now we need to import accounts into metamask, using private keys. 

All done, now we can play with this system. First of all we can request for resources. Because every user can only hold at most 2 resources, you must first release the one you don't need any more, and then you could have place to request other resources.

Unauthorized accounts are not allowed to do request and release operations. But we could add new users into whitelist. Then, new users could participate in this event too.

OK, this is our prototype system. Thank you for watching.