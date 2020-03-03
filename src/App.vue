/* eslint-disable */
<template>
  <div id="app">
    <el-button v-if="!connected" @click="connectMetaMask">Connect to MetaMask</el-button>
    <router-view v-if="connected" :account="account" />
  </div>
</template>

<script>
import UserPage from './components/UserPage'
import AdminPage from './components/AdminPage'

export default {
  components: {
    UserPage,
    AdminPage
  },
  data () {
    return {
      connected: false,
      account: window.ethereum.selectedAddress
    }
  },
  async mounted () {
    if (this.account) {
      this.connected = true
    }

    let that = this
    window.ethereum.on('accountsChanged', function (accounts) {
      console.log(accounts)
      that.account = accounts[0]
      that.connected = true
    })
  },
  methods: {
    async connectMetaMask () {
      try {
        let addrs = await window.ethereum.send({method: 'eth_requestAccounts', param: []})
        this.account = addrs[0]
        this.connected = true
      } catch (error) {
        if (error.code === 4001) { // EIP 1193 userRejectedRequest error
          alert('Please connect to MetaMask.')
        } else if (error.code === -32603) {
          console.log('ethereum method not supported, use another form of calling')
          let addrs = await window.ethereum.send({method: 'eth_requestAccounts'})
          this.account = addrs[0]
          this.connected = true
        } else {
          console.log(error)
        }
      }
    }
  }
}
</script>

<style>

</style>
