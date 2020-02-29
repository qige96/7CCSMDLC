/* eslint-disable */
<template>
  <div>
    <el-row type="flex"  justify="center" align="middle">
      <el-col :span="16">
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
        <el-input type="text" v-model="isWhite">
          <el-button @click="isWhiteListed" slot="append" type="danger">Is Whitelisted?</el-button>
        </el-input>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import { resalocContractInstance } from '../../utils/getContractAndAccount'

export default {
  props: ['account'],
  data () {
    return {
      toAdd: null,
      toRemove: null,
      toAddAdmin: null,
      toRemoveAdmin: null,
      isWhite: null
    }
  },
  methods: {
    addWhitelisted () {
      resalocContractInstance.methods.addWhitelisted(this.toAdd).send({from: this.account})
        .on('confirmation', function (conNum, receipt) {
          console.log(conNum)
        })
        .on('error', function (err) {
          console.log(err)
        })
    },
    removeWhitelisted () {
      console.log('llllllllllllll')
      resalocContractInstance.methods.removeWhitelisted(this.toRemove).send({from: this.account})
        .on('confirmation', function (conNum, receipt) {
          console.log(conNum)
        })
        .on('error', function (err) {
          console.log(err)
        })
    },
    addAdmin () {
      resalocContractInstance.methods.addWhitelisted(this.toAddAdmin).send({from: this.account})
        .on('confirmation', function (conNum, receipt) {
          console.log(conNum)
        })
        .on('error', function (err) {
          console.log(err)
        })
    },
    removeAdmin () {
      resalocContractInstance.methods.removeWhitelisted(this.toRemoveAdmin).send({from: this.account})
        .on('confirmation', function (conNum, receipt) {
          console.log(conNum)
        })
        .on('error', function (err) {
          console.log(err)
        })
    },
    isWhiteListed () {
      resalocContractInstance.methods.isWhitelisted(this.isWhite).call({from: this.account})
        .then(function (result) {
          console.log(result)
          alert(result)
        })
    }
  }
}

</script>

<style>

</style>
