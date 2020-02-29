import Web3 from 'web3'
import contract from 'truffle-contract'

console.log(web3)
const resAloc = require('../build/contracts/SimpleResourceAllocation3.json')
const resalocContract = contract(resAloc)
// resalocContract.setProvider(new Web3.providers.HttpProvider("http://localhost:7545"))
resalocContract.setProvider((new Web3(window.ethereum)).currentProvider)

export { resalocContract }
