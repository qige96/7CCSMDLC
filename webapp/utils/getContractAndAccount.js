import Web3 from 'web3'
import {address, ABI} from './contractConstants'

let web3 = new Web3(window.ethereum)
let resalocContractInstance = new web3.eth.Contract(ABI, address)
let account = window.ethereum.selectedAddress

// console.log(account, window.ethereum.selectedAddress)
console.log(web3)
console.log(resalocContractInstance)

export  { resalocContractInstance, account }

// function getContractInstance () {
//   return new Promise((resolve, reject) => {
//     let web3 = new Web3(window.ethereum)
//     let resalocContractInstance = new web3.eth.Contract(ABI, address)
//     resolve(resalocContractInstance)
//   })
// }

// function getDefaultAccount () {
//   return new Promise((resolve, reject) => {
//     window.ethereum.send({method:'eth_requestAccounts', params:[]})
//       .then(function (accounts) {
//         console.log(accounts)
//         resolve(accounts)
//       })
//       .catch(function (error) {
//         if (error.code === 4001) { // EIP 1193 userRejectedRequest error
//           console.log('Please connect to MetaMask.')
//         } else {
//           console.error(error)
//         }
//       })
//   })
// }

// export {
//   getContractInstance ,
//   getDefaultAccount
// }
