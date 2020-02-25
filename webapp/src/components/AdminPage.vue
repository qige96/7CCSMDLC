/* eslint-disable */
<template>
  <div>
    <el-row type="flex"  justify="center" align="middle">
      <el-col :span="12">
        <h1 style="text-align: center;">ResAloc Admin Page</h1>
        <el-input type="text" v-model="toAdd">
          <el-button @click="addWhitelisted" slot="append">Add Whitelisted</el-button>
        </el-input>
        <el-input type="text" v-model="toRemove">
          <el-button @click="removeWhitelisted" slot="append" type="danger">Del Whitelisted</el-button>
        </el-input>
        <el-input type="text" v-model="toAddAdmin">
          <el-button @click="addAdmin" slot="append">Add Admin</el-button>
        </el-input>
        <el-input type="text" v-model="toRemoveAdmin">
          <el-button @click="removeAdmin" slot="append" type="danger">Del Admin</el-button>
        </el-input>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import { resalocContractInstance, account } from '../../utils/getContractAndAccount'

export default {
  data () {
    return {
      account: account,
      toAdd: null,
      toRemove: null,
      toAddAdmin: null,
      toRemoveAdmin: null
    }
  },
  mounted () {
    let that = this
    window.ethereum.on('accountsChanged', function (accounts) {
      // Time to reload your interface with accounts[0]!
      console.log(accounts)
      that.account = accounts[0]
    })
  },
  methods: {
    addWhitelisted () {
      resalocContractInstance.methods.addWhitelisted(this.toAdd).send(
        { gas: 300000, from: this.account },
        (err, result) => {
          if (err) {
            console.log('err', err)
          } else {
            console.log(result)
          }
        }
      )
    },
    removeWhitelisted () {
      resalocContractInstance.methods.removeWhitelisted(this.toRemove).send(
        { gas: 300000, from: this.account },
        (err, result) => {
          if (err) {
            console.log('err', err)
          } else {
            console.log(result)
          }
        }
      )
    },
    addAdmin () {
      resalocContractInstance.methods.addWhitelisted(this.toAddAdmin).send(
        { gas: 300000, from: this.account },
        (err, result) => {
          if (err) {
            console.log('err', err)
          } else {
            console.log(result)
          }
        }
      )
    },
    removeAdmin () {
      resalocContractInstance.methods.removeWhitelisted(this.toRemoveAdmin).send(
        { gas: 300000, from: this.account },
        (err, result) => {
          if (err) {
            console.log('err', err)
          } else {
            console.log(result)
          }
        }
      )
    }
  }
}

</script>

<style>

</style>
